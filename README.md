# IntentTrade

> **开发阶段 / Work in progress.** 接口与行为可能随时变更，请勿当作生产级交易系统依赖。

用 **LLM 解析 Twitter/X 上 KOL 的交易意图**，在**明确的交易相关推文**上做记录与模拟跟单的研究型工具。

## 界面预览

### 市场意图总览

结构化展示原始推文、交易方向、价格条件、执行状态与模型置信度。

![IntentTrade 市场意图总览](docs/images/dashboard-overview.png)

### 原帖与 AI 解析

滚动查看原帖时，右侧同步展示对应的结构化信号、价格条件与关联模拟成交。

![IntentTrade 原帖与 AI 解析](docs/images/post-analysis.png)

### 标的时间线

把实时 K 线、做多/做空/成交/笔记标记与历史事件流放在同一时间轴上。

![IntentTrade SNDK 标的时间线](docs/images/timeline-sndk.png)

**项目目标**

1. 持续监控指定 KOL 的推文和原图
2. 用大模型抽取：标的、方向、计划/已入场、市价/限价/突破、入场/触发/止损/止盈  
3. 将**结构化喊单**与**非交易闲聊**分开存储  
4. 在价格条件满足时做 **paper 跟单**，并统计 KOL 胜率  
5. 在时间线里用 K 线 + 事件线回看「何时喊单」

**不是什么**

- 不是持牌券商，**没有**真实下单 / 资金划转  
- 不是投资建议；跟单与盈亏模拟仅供研究  
- 当前默认 **paper 执行**；实盘接入明确划在后续阶段

## 许可（禁止商用）

本项目采用 **[PolyForm Noncommercial License 1.0.0](LICENSE)**。

- 允许：个人学习、研究、爱好、非营利教学/科研等非商业用途  
- **禁止**：用于商业产品、收费服务、公司内部营利业务等商业目的  

完整条款见仓库根目录 [`LICENSE`](LICENSE)。若需商业授权，请联系维护者另行约定。

```
Required Notice: Copyright IntentTrade contributors
```

## 当前能力（开发中）

| 模块 | 状态 |
|------|------|
| KOL 拉取（mock / RapidAPI / twitterapi.io / X API） | 可用 |
| LLM 意图解析 + 规则兜底 | 可用 |
| 原图直传识别（正文独立、逐图独立、最终汇总） | 可用 |
| 标的级历史回看（计划调整、成交确认、撤销与旧单取代） | 可用 |
| Agent 工具调用（标的搜索/注册、现价、近期高低点与回撤） | 可用 |
| 结构化信号 vs 描述笔记 / 非交易推文（N/A） | 可用 |
| 价格条件 paper 成交 + SL/TP 结算 | 可用 |
| KOL 胜率统计 | 可用 |
| 看板 + 时间线 K 线（默认 15m） | 可用 |
| 加密：Binance / OKX；美股：Binance **bStock**（如 `SNDK`→`SNDKBUSDT`） | 可用 |
| 实盘经纪商下单 | **未实现** |

## 快速开始

```bash
git clone <your-fork-or-url> IntentTrade
cd IntentTrade
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 或: pip install -e .

cp .env.example .env
# 编辑 .env：填入你自己的 LLM / Twitter 密钥（切勿提交 .env）

# 离线演示（mock 推文 + 样例/行情）
python scripts/run_demo.py

# Web 看板（可开后台 auto_poll）
intent-trade serve --host 127.0.0.1 --port 8787
# 打开 http://127.0.0.1:8787
# 时间线 K 线：http://127.0.0.1:8787/timeline/SNDK
```

常用 CLI：

```bash
intent-trade run
intent-trade report
intent-trade symbol BTC-USD
intent-trade analyze-text "闪迪 1345 上车 止损 1200 目标 1600"
intent-trade reset-db -y
```

## 配置

| 路径 | 说明 |
|------|------|
| `config/settings.yaml` | KOL 列表、twitter 源、行情与 paper 参数 |
| `config/ticker_aliases.yaml` | 标的别名（中英 / 黑话） |
| `.env` | **密钥**（本地，已 gitignore） |
| `.env.example` | 环境变量模板（无密钥） |
| `data/sample/` | mock 推文与价格兜底 |

Twitter 源（`config/settings.yaml` → `twitter.source`）：

- `mock` — 本地样例  
- `rapidapi` / `twitterapi_io` / `x_api` — 需在 `.env` 配置对应 key  

LLM：兼容 Anthropic API 的 `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` / `INTENT_TRADE_LLM_MODEL`。含图片的帖子会自动执行正文识别、逐张原图视觉识别和最终汇总；单图共 3 次调用，多图为 N+2 次。可用 `INTENT_TRADE_VISION_MODEL` 单独指定视觉模型。

历史回看默认检索同一 KOL 最近 7 天、最多 6 条、最多 3 个标的的强相关信号与笔记。后续推文明示改价、已成交、撤销、退出或反向时，旧未成交计划会标记为 `superseded`；已模拟成交记录不会被回滚。图片帖把回看合并进最终汇总调用，因此仍保持单图 3 次调用。

Agent 遇到明确但未知的资产时会搜索并验证 provider symbol，再写入本地 `config/ticker_aliases.learned.yaml`。加密资产使用 `BTC-USD` canonical 格式，美股/ADR 使用 `NVDA` 格式；查价顺序仍为 Binance crypto/bStock 优先，找不到时回退 yfinance。行情工具结果会随分析保存，模型不得把正股高点直接套用为存在溢价的 ADR 入场价。

可选代理（部分地区访问 Binance 主站受限时）：

```env
INTENT_TRADE_HTTP_PROXY=socks5h://user:pass@host:1080
```

## 目录结构

```
src/intent_trade/
  analysis/     # 意图、ticker、多模态
  execution/    # paper 与时机判定
  market/       # 报价 + K 线（Binance/OKX/yfinance/bStock）
  pipeline/     # 拉取 → 解析 → 入库 → 模拟
  storage/      # SQLite
  twitter/      # 社交源
  web/          # FastAPI 看板 + 静态前端
config/
data/sample/
tests/
```

## 安全与隐私

**请勿将真实密钥提交到 Git。**

仓库已忽略（节选）：

- `.env`、各类 `*.pem` / 私钥  
- `Agents.md`、`CLAUDE.md` 等本地/代理说明  
- `.claude/`、`.agents/`、运行库 `data/db/`、缓存与日志  

贡献或 fork 前请自查：

```bash
git status
# 确认没有 .env、Agents.md、真实 token
grep -R "sk-" --include='*.py' --include='*.md' --include='*.yaml' . || true
```

## 风险声明

加密货币与证券交易有重大风险。本软件提供的任何信号、模拟成交或统计**不构成投资建议**。使用本软件产生的任何损失由你自行承担。请遵守你所在司法辖区关于数据抓取、证券与加密交易的法律法规，以及 Twitter/X、交易所 API 的服务条款。

## 路线图（非承诺）

- 更稳的多源社交接入与去重  
- 多图之间的标的关联与冲突诊断增强
- 多 KOL 共识与风控  
- 可选实盘适配器（独立授权与风控模块）  
- 更完整的回测与资金曲线  

## 贡献

欢迎 issue / PR（修复、文档、测试优先）。提交前请：

```bash
source .venv/bin/activate
pytest -q
```

并确认未引入密钥或私有运维文件。

## 致谢

- 行情：Binance 公开数据 / OKX / yfinance  
- 图表：[TradingView Lightweight Charts](https://github.com/tradingview/lightweight-charts)  
- 许可文本：PolyForm Project  

---

**IntentTrade** — 研究 KOL 意图，而不是盲目跟单。
