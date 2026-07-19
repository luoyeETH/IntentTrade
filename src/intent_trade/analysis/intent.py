"""Trading intent extraction — fully AI-driven (LLM extracts symbols + entry/SL/TP)."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from intent_trade.analysis.agent_tools import IntentAgentTools
from intent_trade.analysis.llm_client import (
    chat_json,
    chat_json_agent,
    chat_json_content,
    default_model,
    image_source_from_url,
    llm_enabled,
)
from intent_trade.analysis.ticker_map import TickerMap
from intent_trade.config import AnalysisConfig
from intent_trade.models.domain import (
    Direction,
    EntryMode,
    IntentAction,
    IntentAnalysis,
    PositionState,
    SignalType,
    SocialPost,
)


SYSTEM_PROMPT = """你是专业的交易 KOL 意图结构化引擎。

输入：社媒帖子原文（中/英口语、黑话）。
输出：严格 JSON（不要 markdown、不要解释文字）。

你必须完成：
1) 标的归一：把「大饼/比特币/BTC/闪迪/SNDK…」映射到已知标的库的 canonical symbol。
2) 新别名学习：若出现库中没有的黑话但能确定对应库内标的，写入 alias_learning。
3) 方向：long / short / flat / unknown。
4) 行为 action：
   - open：开新仓；add：加仓；close：平仓；reduce：减仓；
   - hold：继续持有；watch：观察/等待；unknown：无法判断。
5) position_state：planned（计划/条件单）、entered（明确已经买入/持有）、
   exiting（正在退出）、unknown。
6) 入场方式 entry_mode：
   - market：现价/市价/立即执行；limit：到指定价或更优价；
   - stop：突破/跌破指定触发价；range：入场区间；unknown。
   有明确价格但没有“已经买入”语义时，按计划条件单处理，不要当成已成交。
7) 价格字段由原文语义抽取，或由原文明示的相对条件结合工具事实计算（不要编造）：
   - entry_price：开仓/上车/进场/成本/买入价
   - stop_loss：止损
   - take_profit / take_profit_levels：止盈、目标、看到、跌到（按方向理解）
   - trigger_price：条件减仓/平仓/突破/跌破触发价
   - 若原文明确说“从近期高点回撤 50%-70% 时买”，应先查询近期高点，再把它换算为价格区间；evidence 必须注明工具高点和计算过程。
8) signal_type：
   - structured：有标的 + 明确交易行为；开仓/加仓至少有一个明确价格或明确市价语义；
     平仓/减仓即使没有价格也可以是 structured，但不会被当前纸面执行器自动执行。
   - descriptive：情绪、长期观点、无清晰可执行行为。
9) 数字要完整（61000 不要截成 610）；中文「7万」=70000，「1.5k」按上下文理解。
10) Agent 工具：
   - 明确提到一个真实资产但标的库没有时，必须先 search_instruments；找到可信结果后调用 register_instrument 入库，不能直接放弃映射。
   - 帖子引用现价、近期高低点、从高点回撤百分比等外部事实时，调用 get_market_snapshot / get_price_statistics 核验。
   - 工具查到的行情只作为背景事实，不得把“当前价/近期高点”误填成 entry/SL/TP，除非原文明确把它定义为交易价位。
   - 已核验的当前价、高低点、回撤、数据源和时间应写进 summary/descriptive_note/reasoning，让用户能看到工具结论。
   - 搜索已返回名称匹配且行情有效的候选后停止反复搜索同义词，完成必要注册并输出最终结果。
11) 多市场标的：canonical_symbols[0] 必须是当前动作实际针对的主标的；公司正股、ADR、杠杆 ETF、代币化股票要分开。
   正股的高点/回撤不能直接换算成存在溢价的 ADR 入场价。若原文条件基于正股、但实际只能买 ADR 且无法可靠换算，应记录为 descriptive/watch，而不是伪造 ADR 的绝对入场区间。

禁止：
- 凭空捏造价格、标的、方向
- 帖子若是骂战/应援/人生感悟且无交易语义：direction=unknown，signal_type=descriptive，
  canonical_symbols=[]，所有价格 null（不要从图片文件名或臆测 invent 交易计划）
- 忽略已知库已有 symbol 另造代号（除非库中确实没有对应资产）
- 没有明确数字就不要填 entry/stop_loss/take_profit
- “我在1300买入/已上车”是 entered；“1300买/到1300买/挂1300”是 planned + limit。
- “现价买/市价开仓”才是 market；不要因为出现“买入”二字就默认立即成交。
"""

IMAGE_SYSTEM_PROMPT = """你是专业的交易图像分析引擎。你会直接查看原图，独立判断图中是否包含交易意图。

只根据图片本身作答，不参考帖子正文，不从图片 URL 或文件名推断内容。结合图表走势、画线、标注、持仓或订单截图、可见文字和视觉关系，识别标的、方向、行为、入场、止损、止盈和触发条件。看不清或无法确定时必须返回 null/unknown，禁止编造。

输出严格 JSON，不要 markdown 或解释性前缀。
"""

MERGE_SYSTEM_PROMPT = """你是专业的交易 KOL 多模态结果汇总引擎。

输入是正文独立识别结果和逐张原图独立识别结果。你只负责交叉验证、解决冲突并生成一个最终交易意图。不得补充任何阶段中没有依据的标的或价格。正文与图片互补时可以合并；冲突且无法判断时降低置信度并保守输出。输出严格 JSON，不要 markdown 或解释性前缀。
"""

MEMORY_SYSTEM_PROMPT = """你是交易 KOL 的标的级状态记忆与计划变更分析引擎。

输入包含当前推文的独立识别结果，以及同一 KOL 最近的信号和标的笔记。你要判断当前推文与历史计划的关系，并输出当前时点唯一、可执行且不重复的意图状态。历史只能用于理解延续、调整、成交确认、撤销或退出，不能覆盖当前推文明示的事实，也不能把旧价格误当成当前新指令。输出严格 JSON。
"""


def _num(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        x = float(v)
        return x if x > 0 else None
    s = str(v).strip().replace(",", "").replace("$", "").replace("￥", "")
    # 7万 / 1.5万
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*万", s)
    if m:
        return float(m.group(1)) * 10000
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*[kK]", s)
    if m:
        return float(m.group(1)) * 1000
    try:
        x = float(s)
        return x if x > 0 else None
    except ValueError:
        return None


def _enum_value(enum_type, value: Any, default):
    try:
        return enum_type(str(value or default).strip().lower())
    except ValueError:
        return default


def _confidence(value: Any, default: float = 0.0) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


_NUMBER = r"([0-9]+(?:[,.][0-9]+)?\s*(?:万|[kK])?)"


class IntentAnalyzer:
    def __init__(
        self,
        ticker_map: TickerMap,
        config: AnalysisConfig,
        market: Any = None,
    ) -> None:
        self.ticker_map = ticker_map
        self.config = config
        self.agent_tools = (
            IntentAgentTools(
                ticker_map,
                market,
                default_lookback_days=config.agent_price_lookback_days,
            )
            if market is not None and config.agent_tools_enabled
            else None
        )

    def analyze(
        self,
        post: SocialPost,
        *,
        history: Optional[list[dict[str, Any]]] = None,
    ) -> IntentAnalysis:
        mode = (self.config.mode or "llm").lower()
        if mode != "rule_based" and llm_enabled():
            try:
                if self.agent_tools is not None:
                    self.agent_tools.start_session()
                if self._image_urls(post):
                    return self._analyze_multimodal(post, history=history)
                current = self._analyze_llm(post)
                if history and self.config.memory_enabled:
                    return self._review_with_history(post, current, history)
                return current
            except Exception as e:
                fallback = self._analyze_fallback(post)
                fallback.reasoning = f"llm_failed: {e}"
                fallback.analyzer = "fallback"
                return fallback
        return self._analyze_fallback(post)

    def _combined_text(self, post: SocialPost) -> str:
        # Image evidence is handled by native vision calls, never flattened
        # into OCR/caption text and mixed into the post body.
        return (post.text or "").strip()

    @staticmethod
    def _image_urls(post: SocialPost) -> list[str]:
        return list(
            dict.fromkeys(
                url.strip()
                for url in post.media_urls
                if isinstance(url, str)
                and url.strip().lower().startswith(("http://", "https://"))
            )
        )

    def _analyze_llm(
        self,
        post: SocialPost,
        *,
        data_override: Optional[dict[str, Any]] = None,
        analysis_text: Optional[str] = None,
        analyzer: str = "llm",
        stage_details: Optional[dict[str, Any]] = None,
    ) -> IntentAnalysis:
        text = self._combined_text(post) if analysis_text is None else analysis_text
        catalog = self.ticker_map.catalog_for_prompt()
        model = self.config.llm_model or default_model()

        user = f"""已知标的库（canonical symbol + 别名，优先映射到这些 symbol）:
{catalog}

KOL: @{post.author_username}
时间: {post.created_at.isoformat() if post.created_at else ""}
帖子全文:
\"\"\"
{text}
\"\"\"

请只输出一个 JSON 对象，字段：
{{
  "mentions": ["原文称呼"],
  "canonical_symbols": ["库内 symbol，如 BTC-USD / SNDK"],
  "alias_learning": [
    {{"alias": "大饼", "symbol": "BTC-USD", "confidence": 0.95, "reason": "..."}}
  ],
  "direction": "long|short|flat|unknown",
  "action": "open|add|close|reduce|hold|watch|unknown",
  "position_state": "planned|entered|exiting|unknown",
  "entry_mode": "market|limit|stop|range|unknown",
  "signal_type": "structured|descriptive",
  "entry_price": null,
  "entry_price_low": null,
  "entry_price_high": null,
  "trigger_price": null,
  "stop_loss": null,
  "take_profit": null,
  "take_profit_levels": [],
  "entry_condition": "触发条件原文要点",
  "time_horizon": "scalp|day|swing|position|unknown",
  "validity_hours": null,
  "confidence": 0.0,
  "field_confidence": {{"symbol": 0.0, "direction": 0.0, "action": 0.0, "entry": 0.0, "stop_loss": 0.0, "take_profit": 0.0}},
  "evidence": {{"action": "原文证据", "entry": "原文证据"}},
  "summary": "一句话摘要",
  "descriptive_note": "descriptive 时写时间线笔记；structured 可空或补充计划",
  "plan_text": "后续计划/分批/条件单原文要点",
  "reasoning": "简短推理（含如何识别 entry/SL/TP）"
}}

抽取要求（全部由你语义识别）：
- entry_price：开仓/上车/进场/成本/「1345闪迪我上车了」里的 1345 等
- stop_loss：止损/止損/SL
- take_profit：止盈/目标/第一目标；空头「跌到500」可作目标
- trigger_price：例如「跌破61000减仓」「站上1300再买」中的触发价
- 没有就 null，不要猜
- 「7万」→ 70000
- entry_price_low/high 只在原文明确给出价格区间时填写；单点价格不要复制成区间。
- evidence 只引用或短述真实原文依据；无法确认的字段置信度给 0。
"""
        tool_trace: list[dict[str, Any]] = []
        if data_override is not None:
            data = data_override
        elif self.agent_tools is not None:
            data, tool_trace = chat_json_agent(
                SYSTEM_PROMPT,
                user,
                tools=self.agent_tools.definitions(),
                execute_tool=self.agent_tools.execute,
                model=model,
                max_tokens=1600,
                max_rounds=self.config.agent_max_rounds,
            )
        else:
            data = chat_json(SYSTEM_PROMPT, user, model=model, max_tokens=1400)

        # learn aliases into registry
        learned = data.get("alias_learning") or []
        if isinstance(learned, dict):
            learned = [learned]
        for item in learned:
            if not isinstance(item, dict):
                continue
            alias = str(item.get("alias") or "").strip()
            symbol = str(item.get("symbol") or "").strip()
            conf = _confidence(item.get("confidence"))
            if alias and symbol and conf >= 0.6:
                target = self.ticker_map.resolve(symbol) or symbol
                if target in self.ticker_map.by_symbol:
                    self.ticker_map.learn_alias(
                        alias,
                        target,
                        reason=str(item.get("reason") or "llm"),
                        persist=True,
                    )

        raw_mentions = data.get("mentions") or []
        if isinstance(raw_mentions, (str, int, float)):
            raw_mentions = [raw_mentions]
        mentions = [str(x) for x in raw_mentions]
        symbols: list[str] = []
        raw_symbols = data.get("canonical_symbols") or []
        if isinstance(raw_symbols, (str, int, float)):
            raw_symbols = [raw_symbols]
        for c in raw_symbols:
            r = self.ticker_map.resolve(str(c)) or (
                str(c) if str(c) in self.ticker_map.by_symbol else None
            )
            if r and r not in symbols:
                symbols.append(r)
        for m in mentions:
            r = self.ticker_map.resolve(m)
            if r and r not in symbols:
                symbols.append(r)

        direction = _enum_value(Direction, data.get("direction"), Direction.UNKNOWN)
        action = _enum_value(IntentAction, data.get("action"), IntentAction.UNKNOWN)
        position_state = _enum_value(
            PositionState, data.get("position_state"), PositionState.UNKNOWN
        )
        entry_mode = _enum_value(EntryMode, data.get("entry_mode"), EntryMode.UNKNOWN)
        signal_type = _enum_value(
            SignalType, data.get("signal_type"), SignalType.DESCRIPTIVE
        )
        lower_text = text.lower()
        explicit_entered = bool(
            re.search(
                r"(?:我\s*)?(?:已经|已在|已于|已).{0,12}(?:买入|上车|持有|开仓|开多|开空)"
                r"|\b(?:already\s+(?:bought|entered)|i(?:'m| am)\s+holding)\b",
                lower_text,
            )
        ) and not bool(
            re.search(
                r"(?:还没|尚未|未曾|没有|没).{0,6}(?:买入|上车|持有|开仓|开多|开空)",
                lower_text,
            )
        )
        if explicit_entered:
            position_state = PositionState.ENTERED

        entry_f = _num(data.get("entry_price"))
        entry_low = _num(data.get("entry_price_low"))
        entry_high = _num(data.get("entry_price_high"))
        trigger_f = _num(data.get("trigger_price"))
        sl_f = _num(data.get("stop_loss"))
        tp_f = _num(data.get("take_profit"))
        tps_raw = data.get("take_profit_levels") or []
        if isinstance(tps_raw, (str, int, float)):
            tps_raw = [tps_raw]
        tps_f = [x for x in (_num(v) for v in tps_raw) if x is not None]
        if tp_f is None and tps_f:
            tp_f = tps_f[0]
        elif tp_f is not None and tp_f not in tps_f:
            tps_f = [tp_f] + tps_f

        if entry_low is not None or entry_high is not None:
            entry_low = entry_low if entry_low is not None else entry_f
            entry_high = entry_high if entry_high is not None else entry_f
            if entry_low is not None and entry_high is not None:
                entry_low, entry_high = min(entry_low, entry_high), max(entry_low, entry_high)
                if entry_f is None:
                    entry_f = (entry_low + entry_high) / 2
                entry_mode = EntryMode.RANGE

        # A conservative normalization layer keeps malformed but plausible
        # model responses from becoming executable orders.
        if action == IntentAction.UNKNOWN and direction in (Direction.LONG, Direction.SHORT):
            if entry_f is not None or sl_f is not None or tp_f is not None:
                action = IntentAction.OPEN
        if position_state == PositionState.UNKNOWN:
            if action in (IntentAction.CLOSE, IntentAction.REDUCE):
                position_state = PositionState.EXITING
            elif action in (IntentAction.OPEN, IntentAction.ADD):
                position_state = PositionState.PLANNED
        if entry_mode == EntryMode.UNKNOWN and action in (
            IntentAction.OPEN,
            IntentAction.ADD,
        ):
            configured_mode = _enum_value(
                EntryMode,
                self.config.default_entry_mode,
                EntryMode.LIMIT,
            )
            entry_mode = (
                EntryMode.MARKET
                if position_state == PositionState.ENTERED
                else EntryMode.STOP
                if trigger_f is not None and entry_f is None
                else configured_mode
                if entry_f is not None
                else EntryMode.MARKET
            )

        conf = _confidence(data.get("confidence"), 0.5)
        has_level = (
            entry_f is not None
            or sl_f is not None
            or tp_f is not None
            or trigger_f is not None
            or entry_mode == EntryMode.MARKET
        )
        open_actionable = action in (IntentAction.OPEN, IntentAction.ADD)
        exit_actionable = action in (IntentAction.CLOSE, IntentAction.REDUCE)
        if signal_type == SignalType.STRUCTURED:
            if not symbols:
                signal_type = SignalType.DESCRIPTIVE
            elif open_actionable and (
                direction not in (Direction.LONG, Direction.SHORT) or not has_level
            ):
                signal_type = SignalType.DESCRIPTIVE
            elif not open_actionable and not exit_actionable:
                signal_type = SignalType.DESCRIPTIVE
            elif conf < self.config.structured_min_confidence:
                signal_type = SignalType.DESCRIPTIVE

        note = str(data.get("descriptive_note") or "")
        if signal_type == SignalType.DESCRIPTIVE and not note:
            note = str(data.get("summary") or text)[:280]

        return IntentAnalysis(
            post_id=post.id,
            kol_username=post.author_username,
            raw_text=post.text,
            analysis_text=text,
            mentioned_tickers=mentions,
            canonical_symbols=symbols,
            direction=direction,
            action=action,
            position_state=position_state,
            entry_mode=entry_mode,
            signal_type=signal_type,
            entry_price=entry_f,
            entry_price_low=entry_low,
            entry_price_high=entry_high,
            trigger_price=trigger_f,
            stop_loss=sl_f,
            take_profit=tp_f,
            take_profit_levels=tps_f,
            entry_condition=str(data.get("entry_condition") or ""),
            time_horizon=str(data.get("time_horizon") or ""),
            validity_hours=_num(data.get("validity_hours")),
            confidence=conf,
            field_confidence={
                str(k): _confidence(v)
                for k, v in (data.get("field_confidence") or {}).items()
                if isinstance(k, str)
            },
            evidence={
                str(k): str(v)
                for k, v in (data.get("evidence") or {}).items()
                if isinstance(k, str) and v is not None
            },
            summary=str(data.get("summary") or ""),
            descriptive_note=note,
            plan_text=str(data.get("plan_text") or note),
            reasoning=str(data.get("reasoning") or "llm"),
            extracted_fields=stage_details
            or {
                "raw": data,
                "alias_learning": learned,
                "model": model,
                "tool_calls": tool_trace,
            },
            analyzer=("llm_agent" if tool_trace and analyzer == "llm" else analyzer),
        )

    def _analyze_multimodal(
        self,
        post: SocialPost,
        *,
        history: Optional[list[dict[str, Any]]] = None,
    ) -> IntentAnalysis:
        """Analyze text and each original image independently, then reconcile."""
        model = self.config.llm_model or default_model()
        vision_model = os.getenv("INTENT_TRADE_VISION_MODEL") or model
        catalog = self.ticker_map.catalog_for_prompt()
        errors: list[dict[str, Any]] = []

        text_result: Optional[dict[str, Any]] = None
        text_tool_calls: list[dict[str, Any]] = []
        try:
            text_analysis = self._analyze_llm(post)
            text_result = dict(text_analysis.extracted_fields.get("raw") or {})
            text_tool_calls = list(
                text_analysis.extracted_fields.get("tool_calls") or []
            )
        except Exception as exc:
            errors.append({"stage": "text", "error": str(exc)[:240]})

        image_results: list[dict[str, Any]] = []
        for index, url in enumerate(self._image_urls(post), 1):
            prompt = f"""这是本帖第 {index} 张原图。请直接分析图片，而不是把图片转成 OCR 文本。

已知标的库（canonical symbol + 别名，优先映射到这些 symbol）:
{catalog}

只输出一个 JSON 对象，字段：
{{
  "image_index": {index},
  "image_type": "chart|position|order|news|meme|other|unknown",
  "mentions": [],
  "canonical_symbols": [],
  "alias_learning": [],
  "direction": "long|short|flat|unknown",
  "action": "open|add|close|reduce|hold|watch|unknown",
  "position_state": "planned|entered|exiting|unknown",
  "entry_mode": "market|limit|stop|range|unknown",
  "signal_type": "structured|descriptive",
  "entry_price": null,
  "entry_price_low": null,
  "entry_price_high": null,
  "trigger_price": null,
  "stop_loss": null,
  "take_profit": null,
  "take_profit_levels": [],
  "entry_condition": "",
  "time_horizon": "unknown",
  "validity_hours": null,
  "confidence": 0.0,
  "field_confidence": {{}},
  "evidence": {{}},
  "summary": "图片独立结论",
  "descriptive_note": "",
  "plan_text": "",
  "reasoning": "基于哪些视觉证据得出结论"
}}
"""
            try:
                content = [
                    {
                        "type": "image",
                        "source": image_source_from_url(url),
                    },
                    {"type": "text", "text": prompt},
                ]
                result = chat_json_content(
                    IMAGE_SYSTEM_PROMPT,
                    content,
                    model=vision_model,
                    max_tokens=1400,
                )
                result["image_index"] = index
                image_results.append(result)
            except Exception as exc:
                errors.append(
                    {"stage": "image", "image_index": index, "error": str(exc)[:240]}
                )

        if text_result is None and not image_results:
            details = "; ".join(item["error"] for item in errors)
            raise RuntimeError(f"all multimodal analysis stages failed: {details}")

        merge_input = {
            "post": {
                "kol": post.author_username,
                "created_at": post.created_at.isoformat() if post.created_at else "",
            },
            "text_analysis": text_result,
            "image_analyses": image_results,
            "stage_errors": errors,
            "recent_kol_history": history or [],
        }
        merge_prompt = f"""请汇总以下彼此独立的识别结果：
{json.dumps(merge_input, ensure_ascii=False)}

最终只输出一个 JSON 对象，字段必须为：
mentions, canonical_symbols, alias_learning, direction, action, position_state,
entry_mode, signal_type, entry_price, entry_price_low, entry_price_high,
trigger_price, stop_loss, take_profit, take_profit_levels, entry_condition,
time_horizon, validity_hours, confidence, field_confidence, evidence, summary,
descriptive_note, plan_text, reasoning。

若 recent_kol_history 非空，还必须输出：
memory_relation（independent|continues|adjusts|confirms_entry|cancels|exits|reverses|uncertain）、
memory_confidence（0-1）、related_symbol、supersede_signal_ids、memory_summary。

汇总规则：
- 每个价格都要能追溯到正文或某张图片的 evidence；不能把不同标的的价格拼在一起。
- 正文与图片冲突时，在 reasoning 中写明冲突，并选择证据更明确的一方或输出 unknown/null。
- 图片只是行情截图、新闻或表情图且没有交易行为时，不得把它升级成 structured。
- evidence 的值标注来源，例如 text 或 image_1。
- 历史只用于整理当前状态。当前说“已在 1345 上车”时，1345 是当前成交确认，不能继续保留旧计划的 1290-1300 作为当前入场价。
- 仅当当前明确调整、确认成交、撤销、退出或反向时，才列出需要取代的旧未成交 signal id；不得列出已成交信号。
"""

        try:
            final_data = chat_json(
                MERGE_SYSTEM_PROMPT,
                merge_prompt,
                model=model,
                max_tokens=1600,
            )
        except Exception as exc:
            errors.append({"stage": "merge", "error": str(exc)[:240]})
            final_data = text_result or image_results[0]

        memory = self._memory_metadata(final_data, history or [])
        return self._analyze_llm(
            post,
            data_override=final_data,
            analysis_text=self._combined_text(post),
            analyzer="llm_multimodal",
            stage_details={
                "raw": final_data,
                "text_analysis": text_result,
                "image_analyses": image_results,
                "stage_errors": errors,
                "model": model,
                "vision_model": vision_model,
                "tool_calls": text_tool_calls,
                **({"memory": memory} if memory else {}),
            },
        )

    def _review_with_history(
        self,
        post: SocialPost,
        current: IntentAnalysis,
        history: list[dict[str, Any]],
    ) -> IntentAnalysis:
        """Use one post-analysis call to reconcile the current intent with history."""

        current_raw = dict(current.extracted_fields.get("raw") or {})
        payload = {
            "current_post": {
                "id": post.id,
                "created_at": post.created_at.isoformat() if post.created_at else "",
                "text": post.text,
            },
            "current_independent_analysis": current_raw,
            "recent_kol_history": history,
        }
        prompt = f"""请对当前推文做回看性整理：
{json.dumps(payload, ensure_ascii=False)}

输出一个扁平 JSON。首先完整输出当前最终意图字段：
mentions, canonical_symbols, alias_learning, direction, action, position_state,
entry_mode, signal_type, entry_price, entry_price_low, entry_price_high,
trigger_price, stop_loss, take_profit, take_profit_levels, entry_condition,
time_horizon, validity_hours, confidence, field_confidence, evidence, summary,
descriptive_note, plan_text, reasoning。

同时输出记忆字段：
- memory_relation: independent|continues|adjusts|confirms_entry|cancels|exits|reverses|uncertain
- memory_confidence: 0 到 1
- related_symbol: 当前推文实际关联的 canonical symbol
- supersede_signal_ids: 仅列出被当前推文明确定义为失效的旧未成交 signal id
- memory_summary: 一句话说明状态如何从旧计划变化到当前状态

规则：
- 当前推文是主事实，历史只用于消歧和整理状态，不能把旧喊单伪装成当前新喊单。
- “1290-1300 抄底”后说“已在 1345 上车”属于 confirms_entry：当前 entry_price=1345、position_state=entered，旧等待单应列入 supersede_signal_ids。
- “改成 1345 买”属于 adjusts：当前是新的 planned 计划，旧等待单失效。
- “继续等 1290”属于 continues，不应取代仍一致的旧计划，也不要制造重复的新开仓信号。
- 已成交信号绝不能被取代；证据不足时 relation=uncertain 且 supersede_signal_ids=[]。
"""
        try:
            reviewed = chat_json(
                MEMORY_SYSTEM_PROMPT,
                prompt,
                model=self.config.llm_model or default_model(),
                max_tokens=1700,
            )
        except Exception as exc:
            current.extracted_fields["memory"] = {
                "relation": "uncertain",
                "confidence": 0.0,
                "supersede_signal_ids": [],
                "error": str(exc)[:240],
            }
            return current

        memory = self._memory_metadata(reviewed, history)
        return self._analyze_llm(
            post,
            data_override=reviewed,
            analysis_text=self._combined_text(post),
            analyzer="llm_memory",
            stage_details={
                **current.extracted_fields,
                "raw": reviewed,
                "pre_memory_analysis": current_raw,
                "memory": memory,
            },
        )

    def _memory_metadata(
        self,
        data: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not history:
            return {}
        relation = str(data.get("memory_relation") or "uncertain").strip().lower()
        allowed_relations = {
            "independent",
            "continues",
            "adjusts",
            "confirms_entry",
            "cancels",
            "exits",
            "reverses",
            "uncertain",
        }
        if relation not in allowed_relations:
            relation = "uncertain"
        confidence = _confidence(data.get("memory_confidence"), 0.0)
        eligible_ids = {
            str(item.get("signal_id"))
            for item in history
            if item.get("kind") == "signal"
            and item.get("eligible_for_supersede")
            and item.get("signal_id")
        }
        raw_ids = data.get("supersede_signal_ids") or []
        if isinstance(raw_ids, (str, int)):
            raw_ids = [raw_ids]
        can_supersede = relation in {
            "adjusts",
            "confirms_entry",
            "cancels",
            "exits",
            "reverses",
        } and confidence >= self.config.memory_min_confidence
        supersede_ids = (
            [str(value) for value in raw_ids if str(value) in eligible_ids]
            if can_supersede
            else []
        )
        return {
            "relation": relation,
            "confidence": confidence,
            "related_symbol": str(data.get("related_symbol") or ""),
            "supersede_signal_ids": list(dict.fromkeys(supersede_ids)),
            "summary": str(data.get("memory_summary") or ""),
        }

    def _analyze_fallback(self, post: SocialPost) -> IntentAnalysis:
        """Conservative parser for outages and local/offline operation.

        It only emits a structured result when the text contains an explicit
        instrument, direction, action and price/market instruction. It never
        invents stop or target levels, so it is safe as a degraded path.
        """
        text = self._combined_text(post)
        symbols = self.ticker_map.find_in_text(text)
        note = text.replace("\n", " ").strip()
        if len(note) > 280:
            note = note[:277] + "..."

        lower = text.lower()
        long_hit = bool(
            re.search(r"\b(long|bullish|buy|bought)\b|做多|开多|看多|看涨|买入|买|上车|试多|多头", lower)
        )
        short_hit = bool(
            re.search(r"\b(short|bearish|sell|sold)\b|做空|开空|看空|看跌|空头", lower)
        )
        direction = (
            Direction.LONG
            if long_hit and not short_hit
            else Direction.SHORT
            if short_hit and not long_hit
            else Direction.UNKNOWN
        )

        reduce_hit = bool(re.search(r"减仓|减持|reduce|trim", lower))
        close_hit = bool(re.search(r"平仓|清仓|退出|出场|close|exit", lower))
        watch_hit = bool(
            re.search(r"观望|等待|关注|不急|别追|耐心|长期|中期|看好|watch|wait", lower)
        )
        entered_hit = bool(
            re.search(
                r"(?:我|已经|已|当前).{0,10}(?:买入|上车|持有|开仓|开多|开空)|\b(?:bought|holding|entered)\b",
                lower,
            )
        )

        action = IntentAction.UNKNOWN
        if reduce_hit:
            action = IntentAction.REDUCE
        elif close_hit:
            action = IntentAction.CLOSE
        elif direction in (Direction.LONG, Direction.SHORT) and not (
            watch_hit and not re.search(r"开仓|开多|开空|买入|上车|entry|long|short|buy|sell", lower)
        ):
            action = IntentAction.OPEN
        elif watch_hit:
            action = IntentAction.WATCH

        position_state = PositionState.UNKNOWN
        if entered_hit:
            position_state = PositionState.ENTERED
        elif action in (IntentAction.CLOSE, IntentAction.REDUCE):
            position_state = PositionState.EXITING
        elif action in (IntentAction.OPEN, IntentAction.ADD):
            position_state = PositionState.PLANNED

        entry_price = self._fallback_labeled_number(
            text,
            [
                "entry",
                "开仓",
                "开多",
                "开空",
                "进场",
                "入场",
                "@",
            ],
        )
        stop_loss = self._fallback_labeled_number(
            text, ["sl", "stop loss", "stop", "止损", "止損"]
        )
        targets = self._fallback_labeled_numbers(
            text,
            ["tp", "take profit", "target", "目标", "止盈", "看到", "跌到"],
        )
        take_profit = targets[0] if targets else None
        trigger_price = self._fallback_labeled_number(
            text,
            ["跌破", "突破", "站上", "破位", "below", "above", "trigger"],
        )

        # Chinese posts often put the entry number directly before the action:
        # "BTC 62500附近分批做多". Take the first unlabelled number only when
        # the action is otherwise explicit and it is not a stop/target number.
        all_numbers = self._fallback_numbers(text)
        if entry_price is None and action in (IntentAction.OPEN, IntentAction.ADD):
            protected = set(targets)
            if stop_loss is not None:
                protected.add(stop_loss)
            entry_price = next((n for n in all_numbers if n not in protected), None)

        range_match = re.search(
            rf"({_NUMBER[1:-1]})\s*(?:-|~|至|到)\s*({_NUMBER[1:-1]})",
            text,
            re.IGNORECASE,
        )
        entry_low = entry_high = None
        if range_match and re.search(r"入场|进场|entry|附近|接", text, re.IGNORECASE):
            entry_low = _num(range_match.group(1))
            entry_high = _num(range_match.group(2))
            if entry_low is not None and entry_high is not None:
                entry_low, entry_high = min(entry_low, entry_high), max(entry_low, entry_high)
                entry_price = (entry_low + entry_high) / 2

        mode = EntryMode.UNKNOWN
        if entered_hit:
            mode = EntryMode.MARKET
        elif re.search(r"突破|站上|跌破|破位|breakout|break above|break below", lower):
            mode = EntryMode.STOP
        elif entry_low is not None and entry_high is not None:
            mode = EntryMode.RANGE
        elif entry_price is not None and action in (IntentAction.OPEN, IntentAction.ADD):
            mode = _enum_value(
                EntryMode,
                self.config.default_entry_mode,
                EntryMode.LIMIT,
            )
        elif action in (IntentAction.OPEN, IntentAction.ADD):
            mode = EntryMode.MARKET

        explicit_open = action in (IntentAction.OPEN, IntentAction.ADD)
        has_level = (
            entry_price is not None
            or stop_loss is not None
            or take_profit is not None
            or trigger_price is not None
        )
        structured = bool(
            symbols
            and (
                (
                    direction in (Direction.LONG, Direction.SHORT)
                    and explicit_open
                    and (has_level or mode == EntryMode.MARKET)
                )
                or action in (IntentAction.CLOSE, IntentAction.REDUCE)
            )
        )
        confidence = 0.82 if structured else 0.28
        if action == IntentAction.WATCH:
            summary = "观察/等待，没有形成立即跟单指令"
        elif structured:
            summary = f"{action.value} {direction.value} " + (symbols[0] if symbols else "")
        else:
            summary = "规则降级解析：仅记录为描述性观点"

        return IntentAnalysis(
            post_id=post.id,
            kol_username=post.author_username,
            raw_text=post.text,
            analysis_text=text,
            mentioned_tickers=symbols,
            canonical_symbols=symbols,
            direction=direction,
            action=action,
            position_state=position_state,
            entry_mode=mode,
            signal_type=SignalType.STRUCTURED if structured else SignalType.DESCRIPTIVE,
            entry_price=entry_price,
            entry_price_low=entry_low,
            entry_price_high=entry_high,
            trigger_price=trigger_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            take_profit_levels=targets,
            entry_condition=(
                "突破/跌破触发" if mode == EntryMode.STOP else ""
            ),
            confidence=confidence,
            field_confidence={
                "symbol": 0.9 if symbols else 0.0,
                "direction": 0.85 if direction != Direction.UNKNOWN else 0.0,
                "action": 0.82 if action != IntentAction.UNKNOWN else 0.0,
                "entry": 0.8 if entry_price is not None else 0.0,
                "stop_loss": 0.8 if stop_loss is not None else 0.0,
                "take_profit": 0.8 if take_profit is not None else 0.0,
                "trigger_price": 0.8 if trigger_price is not None else 0.0,
            },
            evidence={"raw": note[:240]},
            summary=summary,
            descriptive_note=note if not structured else "",
            plan_text=note,
            reasoning="conservative_rule_fallback",
            analyzer="rule_based",
        )

    @staticmethod
    def _fallback_numbers(text: str) -> list[float]:
        return [
            value
            for m in re.finditer(_NUMBER, text, re.IGNORECASE)
            if (value := _num(m.group(1))) is not None
        ]

    @staticmethod
    def _fallback_labeled_numbers(text: str, labels: list[str]) -> list[float]:
        if not text:
            return []
        patterns: list[str] = []
        for label in sorted(labels, key=len, reverse=True):
            escaped = re.escape(label)
            if re.fullmatch(r"[A-Za-z0-9 ]+", label):
                patterns.append(rf"(?<![A-Za-z]){escaped}(?![A-Za-z])")
            else:
                patterns.append(escaped)
        label_re = "|".join(patterns)
        values: list[float] = []
        for match in re.finditer(label_re, text, re.IGNORECASE):
            tail = text[match.end() : match.end() + 32]
            numbers = list(re.finditer(_NUMBER, tail, re.IGNORECASE))
            if not numbers:
                continue
            number = numbers[0]
            # TP1/TP2 and "第一目标" contain an ordinal before the actual
            # price. Ignore a small standalone ordinal when another number
            # follows it.
            ordinal = _num(number.group(1))
            if (
                ordinal is not None
                and ordinal <= 10
                and re.fullmatch(r"\s*\d+\s*", number.group(1))
                and len(numbers) > 1
            ):
                number = numbers[1]
            value = _num(number.group(1))
            if value is not None and value not in values:
                values.append(value)
        return values

    @classmethod
    def _fallback_labeled_number(cls, text: str, labels: list[str]) -> Optional[float]:
        values = cls._fallback_labeled_numbers(text, labels)
        return values[0] if values else None
