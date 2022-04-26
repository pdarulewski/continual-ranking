import glob
import json
import logging
import os
from typing import List, Tuple, Dict

import hydra
from continual_ranking.dpr.data.qa_validation import calculate_matches
from continual_ranking.dpr.utils.model_utils import get_model_obj, load_states_from_checkpoint
from omegaconf import DictConfig, OmegaConf

from continual_ranking.config.paths import DATA_DIR
from continual_ranking.dpr.indexer.faiss_indexers import LocalFaissRetriever
from continual_ranking.dpr.models import get_bert_biencoder_components
from continual_ranking.dpr.utils.options import setup_logger, setup_cfg_gpu, set_cfg_params_from_state

logger = logging.getLogger()
setup_logger(logger)


def validate(
        passages: Dict[object, Tuple[str, str]],
        answers: List[List[str]],
        result_ctx_ids: List[Tuple[List[object], List[float]]],
        workers_num: int,
        match_type: str,
) -> List[List[bool]]:
    logger.info("validating passages. size=%d", len(passages))
    match_stats = calculate_matches(passages, answers, result_ctx_ids, workers_num, match_type)
    top_k_hits = match_stats.top_k_hits

    logger.info("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info("Validation results: top k documents hits accuracy %s", top_k_hits)
    return match_stats.questions_doc_hits


def save_results(
        passages: Dict[object, Tuple[str, str]],
        questions: List[str],
        answers: List[List[str]],
        top_passages_and_scores: List[Tuple[List[object], List[float]]],
        per_question_hits: List[List[bool]],
        out_file: str,
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    # assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        results_item = {
            "question": q,
            "answers":  q_answers,
            "ctxs":     [
                {
                    "id":         results_and_scores[0][c],
                    "title":      docs[c][1],
                    "text":       docs[c][0],
                    "score":      scores[c],
                    "has_answer": hits[c],
                }
                for c in range(ctxs_num)
            ],
        }

        # if questions_extra_attr and questions_extra:
        #    extra = questions_extra[i]
        #    results_item[questions_extra_attr] = extra

        merged_data.append(results_item)

    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info("Saved results * scores  to %s", out_file)


def get_all_passages(ctx_sources):
    all_passages = {}
    for ctx_src in ctx_sources:
        ctx_src.load_data_to(all_passages)
        logger.info("Loaded ctx data: %d", len(all_passages))

    if len(all_passages) == 0:
        raise RuntimeError("No passages data found. Please specify ctx_file param properly.")
    return all_passages


@hydra.main(config_path="../../config", config_name="evaluation")
def main(cfg: DictConfig):
    cfg = setup_cfg_gpu(cfg)
    saved_state = load_states_from_checkpoint(cfg.model_file)

    set_cfg_params_from_state(saved_state.encoder_params, cfg)

    logger.info("CFG (after gpu  configuration):")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    tensorizer, encoder, _ = get_bert_biencoder_components(cfg, inference_only=True)

    logger.info("Loading saved model state ...")
    encoder.load_state(saved_state, strict=False)

    encoder_path = cfg.encoder_path
    if encoder_path:
        logger.info("Selecting encoder: %s", encoder_path)
        encoder = getattr(encoder, encoder_path)
    else:
        logger.info("Selecting standard question encoder")
        encoder = encoder.question_model

    encoder.eval()

    model_to_load = get_model_obj(encoder)
    vector_size = model_to_load.config.hidden_size
    logger.info("Encoder vector_size=%d", vector_size)

    # get questions & answers
    questions = []
    questions_text = []
    question_answers = []

    if not cfg.qa_dataset:
        logger.warning("Please specify qa_dataset to use")
        return

    ds_key = cfg.qa_dataset
    logger.info("qa_dataset: %s", ds_key)

    qa_src = hydra.utils.instantiate(cfg.datasets[ds_key])
    qa_src.load_data()

    total_queries = len(qa_src)
    for i in range(total_queries):
        qa_sample = qa_src[i]
        question, answers = qa_sample.query, qa_sample.answers
        questions.append(question)
        question_answers.append(answers)

    logger.info("questions len %d", len(questions))
    logger.info("questions_text len %d", len(questions_text))

    index = hydra.utils.instantiate(cfg.indexers[cfg.indexer])
    logger.info("Local Index class %s ", type(index))
    index_buffer_sz = index.buffer_size
    index.init_index(vector_size)
    retriever = LocalFaissRetriever(encoder, cfg.batch_size, tensorizer, index)

    logger.info("Using special token %s", qa_src.special_query_token)
    questions_tensor = retriever.generate_question_vectors(questions, query_token=qa_src.special_query_token)

    if qa_src.selector:
        logger.info("Using custom representation token selector")
        retriever.selector = qa_src.selector

    index_path = cfg.index_path
    id_prefixes = []
    ctx_sources = []
    if index_path and index.index_exists(index_path):
        logger.info("Index path: %s", index_path)
        retriever.index.deserialize(index_path)
    else:
        # send data for indexing
        for ctx_src in cfg.ctx_datatsets:
            ctx_src = hydra.utils.instantiate(cfg.ctx_sources[ctx_src])
            id_prefixes.append(ctx_src.id_prefix)
            ctx_sources.append(ctx_src)
            logger.info("ctx_sources: %s", type(ctx_src))

        logger.info("id_prefixes per dataset: %s", id_prefixes)

        # index all passages
        ctx_files_patterns = cfg.encoded_ctx_files

        logger.info("ctx_files_patterns: %s", ctx_files_patterns)
        if ctx_files_patterns:
            assert len(ctx_files_patterns) == len(id_prefixes), "ctx len={} pref leb={}".format(
                len(ctx_files_patterns), len(id_prefixes)
            )
        else:
            assert (
                    index_path or cfg.rpc_index_id
            ), "Either encoded_ctx_files or index_path pr rpc_index_id parameter should be set."

        input_paths = []
        path_id_prefixes = []
        for i, pattern in enumerate(ctx_files_patterns):
            pattern_files = glob.glob(os.path.join(DATA_DIR, pattern))
            input_paths.extend(pattern_files)
            pattern_id_prefix = id_prefixes[i]
            path_id_prefixes.extend([pattern_id_prefix] * len(pattern_files))
        logger.info("Embeddings files id prefixes: %s", path_id_prefixes)
        logger.info("Reading all passages data from files: %s", input_paths)
        retriever.index_encoded_data(input_paths, index_buffer_sz, path_id_prefixes=path_id_prefixes)
        if index_path:
            retriever.index.serialize(index_path)

    # get top k results
    top_results_and_scores = retriever.get_top_docs(questions_tensor.numpy(), cfg.n_docs)

    all_passages = get_all_passages(ctx_sources)
    questions_doc_hits = validate(
        all_passages,
        question_answers,
        top_results_and_scores,
        cfg.validation_workers,
        cfg.match,
    )

    if cfg.out_file:
        save_results(
            all_passages,
            questions_text if questions_text else questions,
            question_answers,
            top_results_and_scores,
            questions_doc_hits,
            cfg.out_file,
        )


if __name__ == "__main__":
    main()
