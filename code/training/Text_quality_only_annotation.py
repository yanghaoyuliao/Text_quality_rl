import re
import logging
from typing import List, Dict, Tuple, Optional
import torch
try:
    from swift.plugin import ORM, orms
except ImportError:
    from swift.rewards import ORM, orms
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RELEVANCE_MODEL = "/usr/data/wjx/trove/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class TextQualityRewardModel(ORM):
    """
    句子级相似度奖励模型。
    """

    def __init__(self, device: str = "cuda", args=None, **kwargs):
        # Swift instantiates ORM rewards with an ``args`` keyword.
        del args, kwargs
        self.device = device

        self.tuple_pattern_3_groups_str = r'\(\"(\d+)\"\,\s*\"(\d+)\"\,\s*\"([^"]*)\"\)'
        self.tuple_extractor_3_groups = re.compile(self.tuple_pattern_3_groups_str)

        self._init_evaluators()


    def _init_evaluators(self):
        """初始化相关性评估模型和ROUGE评估器"""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        logger.info("ROUGE-L 评估器加载成功。")

        try:
            # 引入所有可能涉及的模块
            import transformers.modeling_utils
            import transformers.integrations
            import deepspeed

            # --- 核心修改：备份原始函数 ---
            # 我们需要备份 modeling_utils 里的引用，这才是 AutoModel 真正调用的地方
            _orig_check_utils = getattr(transformers.modeling_utils, "is_deepspeed_zero3_enabled", None)
            _orig_check_integ = getattr(transformers.integrations, "is_deepspeed_zero3_enabled", None)

            # 定义一个永远返回 False 的函数
            def _force_false(): return False

            # --- 实施 Patch (欺骗 Transformers) ---
            # 1. 覆盖 modeling_utils 中的副本 (关键！)
            if _orig_check_utils:
                transformers.modeling_utils.is_deepspeed_zero3_enabled = _force_false

            # 2. 覆盖 integrations 中的原版 (以防万一)
            if _orig_check_integ:
                transformers.integrations.is_deepspeed_zero3_enabled = _force_false

            logger.info("🔒 [SimilarityReward] 已通过双重 Monkey Patch 强制屏蔽 Zero-3 检测...")

            try:
                # 3. 双重保险：使用 DeepSpeed 上下文禁用 Init
                with deepspeed.zero.Init(enabled=False):
                    # 4. 显式加载到 CPU，防止任何 CUDA 显存分配干扰
                    self.relevance_model = SentenceTransformer(RELEVANCE_MODEL, device="cpu")

                # 5. 手动移动到正确的 GPU 设备并设为评估模式
                self.relevance_model.to(self.device)
                self.relevance_model.eval()

                logger.info(f"✅ 相关性模型加载成功 (Device: {self.device})。")

            finally:
                # --- 恢复现场 (至关重要) ---
                # 必须恢复原始函数，否则后续的主模型训练会因为检测不到 Zero-3 而 OOM
                if _orig_check_utils:
                    transformers.modeling_utils.is_deepspeed_zero3_enabled = _orig_check_utils
                if _orig_check_integ:
                    transformers.integrations.is_deepspeed_zero3_enabled = _orig_check_integ
                logger.info("🔓 [SimilarityReward] 已恢复 DeepSpeed Zero-3 环境检测。")

        except Exception as e:
            logger.error(f"❌ 加载相关性模型失败: {RELEVANCE_MODEL}")
            logger.error(f"错误详情: {e}")
            raise

    def get_content_after_think_tag(self, text: str) -> str:
        """获取 '</think>' 标签之后的内容。"""
        parts = text.split('</think>', 1)
        if len(parts) > 1:
            return parts[1].strip()
        else:
            return text.strip()

    def extract_annotation(self, text: str) -> Tuple[str, bool]:
        """Extract the Annotation section and report whether its header exists."""
        content = self.get_content_after_think_tag(text)
        match = re.search(r"##\s*Annotation\s*:\s*", content, flags=re.IGNORECASE)
        if match is None:
            return content.strip(), False
        content = content[match.end():]
        reason = re.search(r"##\s*Reason\s*:\s*", content, flags=re.IGNORECASE)
        if reason is not None:
            content = content[:reason.start()]
        return content.strip(), True

    def split_by_sent_tag(self, text: str) -> List[str]:
        """Split the current Annotation format into <SENT> blocks."""
        sentences = [item.strip() for item in re.findall(
            r"<SENT\b[^>]*>(.*?)</SENT>", text, flags=re.IGNORECASE | re.DOTALL
        ) if item.strip()]
        return sentences

    def has_valid_sent_tags(self, text: str) -> bool:
        """Require balanced SENT tags and no unwrapped text in Annotation."""
        opening = re.findall(r"<SENT\b[^>]*>", text, flags=re.IGNORECASE)
        closing = re.findall(r"</SENT\s*>", text, flags=re.IGNORECASE)
        if not opening or len(opening) != len(closing):
            return False
        blocks = list(re.finditer(
            r"<SENT\b[^>]*>(.*?)</SENT\s*>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ))
        if len(blocks) != len(opening) or any(not match.group(1).strip() for match in blocks):
            return False
        unwrapped = re.sub(
            r"<SENT\b[^>]*>.*?</SENT\s*>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return not unwrapped.strip()

    def parse_generation(self, text: str) -> Tuple[Optional[str], Optional[List[Dict]]]:
        """解析包含多个[PROVE]标记的文本，提取所有文本和引文"""
        generated_text = re.sub(r'\[PROVE:\s*.*?\]', '', text, flags=re.DOTALL).strip()
        citations = []
        tuple_pat = self.tuple_pattern_3_groups_str
        citation_pattern = r'\[PROVE:\s*(' + tuple_pat + r'(?:\s*,\s*' + tuple_pat + r')*)\s*\]'

        for match in re.finditer(citation_pattern, text, flags=re.DOTALL):
            citation_block = match.group(1)
            for tuple_match in re.finditer(self.tuple_extractor_3_groups, citation_block):
                citations.append({
                    "doc_id": int(tuple_match.group(1)),
                    "sent_id": int(tuple_match.group(2)),
                    "relation": tuple_match.group(3)
                })
        return generated_text, citations

    def _compute_rouge_score(self, reference: str, generated: str) -> float:
        """计算ROUGE-L分数"""
        if not reference or not generated:
            return 0.0
        scores = self.rouge_scorer.score(reference, generated)
        return scores['rougeL'].fmeasure

    def _sentence_coverage(
            self,
            generated_sentences: List[str],
            reference_sentences: List[str],
    ) -> float:
        """Penalize outputs that cover fewer sentences than the reference."""
        if reference_sentences:
            return min(1.0, len(generated_sentences) / len(reference_sentences))
        return 1.0

    def __call__(
            self,
            completions: List[str],
            annotation_references: List[str],
            **kwargs
    ) -> List[float]:
        """
        主函数 - 从参考段出发计算均值
        """
        rewards = []

        for completion_text, reference_text in zip(completions, annotation_references):

            gen_content, has_annotation_header = self.extract_annotation(completion_text)
            logger.info(f"\n\t⭐ --- ⭐ --- ⭐ ---😺 当前处理文本：\n\t{gen_content}\n\t⭐ --- ⭐ --- ⭐ ---🐼对应标准文本：\n\t{reference_text}\n\t☕ --- ☕ --- ☕ ---☕句子级匹配开始✈️")
            ref_content, _ = self.extract_annotation(reference_text)
            gen_sentences = self.split_by_sent_tag(gen_content)
            ref_sentences = self.split_by_sent_tag(ref_content)
            valid_sent_tags = self.has_valid_sent_tags(gen_content)

            total_sentence_rouge = 0

            if not has_annotation_header:
                rewards.append(0.0)
                logger.warning("❌ 缺少 ## Annotation: 标志，格式奖励为 0.0")
                continue

            if not valid_sent_tags:
                rewards.append(0.0)
                logger.warning("❌ <SENT> 标签格式错误，格式奖励为 0.0")
                continue

            if not gen_sentences:
                rewards.append(0.0)
                logger.warning("❌ 没有生成任何可评估的句子，奖励分数为 0.0")
                continue

            for gen_sentence in gen_sentences:
                gen_pure_text, _gen_citations = self.parse_generation(gen_sentence)

                max_sim_score = -1
                sentence_rouge_score = 0.0
                best_ref_sentence_text = ""

                if ref_sentences:
                    ref_pure_texts = [self.parse_generation(s)[0] or "" for s in ref_sentences]

                    if gen_pure_text and any(ref_pure_texts):
                        gen_embedding = self.relevance_model.encode(gen_pure_text, convert_to_tensor=True)
                        ref_embeddings = self.relevance_model.encode(ref_pure_texts, convert_to_tensor=True)

                        cosine_scores = util.pytorch_cos_sim(gen_embedding, ref_embeddings).squeeze()

                        if cosine_scores.numel() > 0:
                            if cosine_scores.dim() == 0:
                                max_sim_idx = 0
                                max_sim_score = cosine_scores.item()
                            else:
                                max_sim_idx = torch.argmax(cosine_scores).item()
                                max_sim_score = cosine_scores[max_sim_idx].item()

                            best_ref_sentence = ref_sentences[max_sim_idx]
                            best_ref_sentence_text, _ = self.parse_generation(best_ref_sentence)

                if max_sim_score < 0.45:
                    sentence_rouge_score = 0.0
                    logger.warning(
                        f"\n\t❌ 句子:\n\t{gen_pure_text}\n\t未找到匹配项 (最高 {max_sim_score:.2f})，奖励分数记为 0。")
                else:
                    sentence_rouge_score = self._compute_rouge_score(best_ref_sentence_text, gen_pure_text)
                    logger.info(
                        f"\n\t⭐ 生成句子:\n\t'{gen_pure_text}'\n\t✅匹配到参考句子:\n\t {best_ref_sentence_text} (Sim: {max_sim_score:.2f})，奖励分数: {sentence_rouge_score:.2f}")

                total_sentence_rouge += sentence_rouge_score

            avg_sentence_rouge_reward = total_sentence_rouge / len(gen_sentences)
            sentence_coverage = self._sentence_coverage(
                gen_sentences,
                ref_sentences,
            )
            reward = avg_sentence_rouge_reward * sentence_coverage
            rewards.append(reward)
            logger.info(f"\n\t🐦 --- 🐦 --- 🐦 ---😺 当前处理文本的奖励值为:\n\t{reward}")
            if sentence_coverage < 1.0:
                logger.info(
                    "\t📏 句子覆盖率惩罚: sentence_coverage=%.3f",
                    sentence_coverage,
                )


        logger.info(f"\n\t🎉 基于句子匹配的相似度奖励计算完成, 奖励列表: {[round(r, 3) for r in rewards]}")
        return rewards

orms['text_quality_reward'] = TextQualityRewardModel
