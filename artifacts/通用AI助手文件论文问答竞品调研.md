# 通用 AI 助手在「文件 / 论文问答」场景的产品设计调研报告

> 调研目的：为自研求职项目「论文阅读问答助手」（DeepSeek LLM + Ollama bge-m3 embedding + Chroma 向量库 + 自实现 BM25 + RRF 混合检索 + 引用溯源 [n]）做竞品借鉴。
> 调研范围：仅覆盖**通用 AI 助手**这一类——ChatGPT、Claude、Google NotebookLM、豆包 / Kimi / 通义千问。不覆盖专业论文 SaaS、开源项目、学术生态工具。
> 调研时间：2026-10-08。
> 标注约定：
> - **【已核验】** = 来自官方文档 / 帮助中心 / 官方博客，附来源 URL。
> - **【二手报道】** = 媒体评测 / 第三方博客，非官方口径。
> - **【推断】** = 无直接官方证据，从产品行为或 API 结构推测，不得当作事实引用。

---

## 一、Google NotebookLM（重点）

NotebookLM 是这一组里**唯一把"引用溯源"当作产品核心来设计**的通用助手，也是个人项目最值得对标的对象。

### 1. 核心文件 / 论文问答功能
- **【已核验】支持来源格式极广**：Google Docs / Slides、PDF、Microsoft Word（docx）、txt、Markdown、CSV、PowerPoint（pptx）、ePub、网页 URL、公开 YouTube 视频（仅用字幕文本）、音频文件；每个来源最多 50 万词或 200MB。
  来源：https://support.google.com/notebooklm/answer/16215270?hl=en ；https://workspace.google.com/intl/en_sg/products/gemini-notebook/
- **【已核验】单笔记本多来源**：免费版每个笔记本最多 50 个来源，可跨多份文档检索聚合回答；每个来源前有**复选框**，可临时勾选 / 取消勾选本次问答纳入哪些来源。
  来源：https://edu.google.com/intl/ALL_us/ai-notebooklm/ ；https://support.google.com/notebooklm/answer/16179559?hl=en
- **【已核验】问答被"锁死在来源内"**：官方明确"Chat responses in NotebookLM only use data from your sources"，若问一个来源外的创意问题会直接回 "NotebookLM can't answer this question"。这是与 ChatGPT 最大的产品哲学差异——**宁可拒答，也不脱离来源瞎编**。

### 2. 文档管理与交互设计
- **【已核验】以"笔记本（Notebook）"为项目容器**：一个笔记本 = 一组来源 + 聊天 + 笔记 + Studio 产物；免费版最多 100 个笔记本。
- **【已核验】Noteboard（记事板）**：可把聊天里的回答、来源里的片段、自己写的笔记"钉"在板子上，作为跨文档的知识拼贴。
  来源：https://blog.google/innovation-and-ai/products/notebooklm-new-features-availability/
- **【已核验】聊天可配置风格**：Default / Learning Guide（教学模式）/ Custom（自定义人设），可选回答长短；回答可"Save to note"钉到记事板，**保存后表格和可点击的行内引用一起保留**。
  来源：https://support.google.com/notebooklm/answer/16179559?hl=en

### 3. 引用溯源设计（最重要）
- **【已核验】引用 = 直接引文（direct quote）**：NotebookLM 直接从来源里摘原话当作引用，而不是只给个文件名。
- **【已核验】悬停（hover）→ 预览完整引文**："You can hover over any citation to get the full quoted text right away."
- **【已核验】点击（select）→ 自动跳转到原文位置**："If you select a citation, NotebookLM automatically navigates to the location of the quote, so you can easily view it in context."
  以上三条来源：https://support.google.com/notebooklm/answer/16179559?hl=en
- **【二手报道】引用呈现为"灰色椭圆里的小数字"**：DataCamp 实测描述为"little numbers in grey ovals next to the information"，点击数字 3 后左侧"Source guide"面板打开并定位到对应段落。
  来源：https://www.datacamp.com/tutorial/notebooklm
- **【二手报道】引用粒度是 chunk 级而非整篇文档级**："Citations are at the chunk level, not the source level... clicking it highlights the exact passage in the source."
  来源：https://learn.techwithdarin.com/guides/notebooklm/
- **【二手报道】存在"引用漂移（citation drift）"风险**：数字看着像那么回事，但指向的段落可能只是"相关"而非"真正支撑这句话"——这是对个人项目的直接警示。
  来源：https://notebooklm-to-pdf.com/fa/blog/notebooklm-citations-accuracy

### 4. 多文档问答与跨文档能力
- **【已核验】支持跨多来源聚合回答**，且回答中每一条主张都挂 chunk 级引用；免费版每日 50 次聊天查询上限。
- **【推断】引用数不做硬上限**，按主张密度自然生成（官方文档未写死引用条数限制）。

### 5. 架构线索
- **【已核验】底层用 Gemini 多模态**，官方博客称其用 Gemini 1.5 的多模态能力在来源间"建立联系"。
- **【推断】是 RAG（检索增强）+ chunk 级索引**：引用能定位到"原话片段"，必然是先切块再检索、生成时把 chunk id 带回；但官方不披露具体检索算法（是否向量 + 关键词混合、是否 rerank 均未公开）。
- **【已核验】Agent 化是高阶付费能力**：Google AI Ultra 订阅者可在聊天里联网搜索、跑代码、生成可下载文件 / 图表 / 图片，即"agentic capabilities"。
  来源：https://support.google.com/notebooklm/answer/16179559?hl=en

### 6. 写作 / 笔记辅助能力
- **【已核验】Audio Overview（音频概览 / 类播客）**：两个 AI 主持人把一组来源聊成一段播客式对谈，免费版每日 3 次；另有视频概览、信息图。
- **【已核验】Studio 产物可导出**：笔记可导出到 Google Docs / Sheets；新版本可生成 PDF、docx、Markdown、xlsx、pptx、图表（png/svg）等直接下载。
  来源：https://support.google.com/notebooklm/answer/16262519?hl=en ；https://blog.google/innovation-and-ai/products/notebooklm/better-research-notebooklm/
- **【已核验】笔记可"Convert to source"**：把自己写的笔记反过来当成新来源喂给模型——来源与笔记双向流动。

### 7. 免费 / 付费门槛
- **【二手报道】免费版**：100 笔记本 / 50 来源 / 50 次问答每天 / 3 次音频每天。
- **【二手报道】付费**：Google AI Plus 约 $4.99/月、Google AI Pro 约 $19.99/月，更高档（Ultra）单笔记本来源上限提到 300–600。
  来源：https://toolchase.com/tool/notebooklm/ ；https://companyview.io/software/notebooklm

### 8. 对个人项目的可借鉴程度：**高**
一句话：**它的"灰色数字引用 → 悬停看原话 → 点击跳原文高亮"就是你 [n] 引用溯源应该长成的样子。**

可直接搬进个人项目的点：
1. **悬停卡片显示"原话引文"**，而不只是文件名 —— 你的 [n] 点击后应弹出该 chunk 原文。
2. **点击引用后在原文区高亮划线定位**（NotebookLM 的"navigates to the location of the quote"）——前端做一个左右分栏：左聊天、右原文，点击 [n] 右栏滚动并高亮该段。
3. **来源复选框**：上传多篇论文后，可勾选本次只让模型用其中几篇——个人项目加个 checkbox 成本极低，但演示效果很好。

---

## 二、ChatGPT（附件 / Projects 问答）

ChatGPT 的附件问答是"长上下文硬塞"路线，**故意不做精细引用锚点**，与 NotebookLM 形成鲜明对照。

### 1. 核心文件 / 论文问答功能
- **【已核验】支持格式**：PDF、DOCX、XLSX、CSV、TXT、JSON、HTML、图片（JPEG/PNG/GIF/WebP）、代码文件、音频转写。
  来源：https://help.openai.com/zh-hans-cn/articles/20001052-using-library-to-manage-files-in-chatgpt
- **【已核验】单条消息最多 20 个文件**（此前为 10 个，2026 年扩容）；文本 / 文档类单文件上限 200 万 token，电子表格约 50MB，图片 20MB，单文件总大小 512MB。
  来源：https://help.openai.com/zh-hans-cn/articles/6825453-chatgpt-release-notes ；OpenAI Help「Using Library to manage files」

### 2. 文档管理与交互设计
- **【已核验】Projects（项目）**：一个项目里挂参考文件 + 自定义指令，项目内所有对话都能复用这些文件。
  来源：OpenAI Help「Projects in ChatGPT」（多语言版本）
- **【已核验】Library（文件库）**：聊天里传过的文件自动归档到独立资料库，跨对话复用；资料库额度与每日附件额度分开算。
- **【二手报道】项目文件数按档分**：Free 5 个 / 项目，Go、Plus 25 个，Pro、Business、Enterprise 40 个；文件从 Library 里挂入。
  来源：https://www.ai-toolbox.co/chatgpt-management-and-productivity/how-to-use-chatgpt-projects-guide-2026

### 3. 引用溯源设计（最重要）
- **【二手报道】附件问答没有原生行内引用角标、不自动给页码**："ChatGPT 并非专为 PDF 设计：它不支持自动引用页码"；"Uploading a PDF runs a text extraction step first, and page breaks do not survive it"——即 PDF 抽文本后分页信息丢失，模型给的页码常是猜的。
  来源：https://cloud.tencent.cn/developer/article/2532558 ；https://www.fileuploadgpt.com/blog/chatgpt-cite-page-numbers
- **【二手报道】要引用只能靠提示词硬逼**：用户得在提问里写"Quote the exact sentence and give the page number"，模型才会回引文；且需点名文件名（"From q1_2026.pdf, …"）才会真去用那份文件。
  来源：https://aitoolsguidebook.com/en/articles/chatgpt-project-files-not-referenced/
- **【推断】其联网搜索（web search）结果会显示编号式角标链接**，但这是网页来源、不是上传文档的引用；**附件问答本身不做 chunk 级锚点**。

### 4. 多文档问答与跨文档能力
- **【已核验】单消息 20 文件、项目内多文件可被同会话引用**，支持跨文件比较。
- **【二手报道】但因无引用锚点，多文档时用户难以核对"这句话到底出自哪篇哪段"**。

### 5. 架构线索
- **【推断】以长上下文（long context）吞文件为主**：抽文本 → 塞进上下文，而非为文档单独建可检索 chunk 索引；Projects 更多是"指令 + 文件挂载"的上下文组织。
- **【已核验】有 Memory（记忆）**跨会话保存用户偏好；付费版有 App Connectors（Drive、Slack、GitHub 等 60+ 连接器）。

### 6. 写作 / 笔记辅助能力
- **【二手报道】**主要靠对话内生成 + Library 归档，无 NotebookLM 那种一键播客 / 记事板；偏通用助手而非阅读笔记工具。

### 7. 免费 / 付费门槛
- **【二手报道】Free**：每天约 3 次文件上传，项目 5 文件，Library 500MB。
- **【二手报道】Plus $20/月**：约每 3 小时 80 文件，Library 20GB，项目 25 文件。
  来源：https://www.fileuploadgpt.com/blog/chatgpt-file-upload-limit-free-plan

### 8. 对个人项目的可借鉴程度：**低（作为"反面教材"反而有价值）**
一句话：**它的"无引用"恰恰证明你的 [n] 溯源是差异化亮点，不用学它。**

可借鉴点（仅 1 条）：
1. **Library 文件库**的思路值得借：个人项目可做一个"已上传论文列表"，上传一次、多轮问答复用，而不是每轮重传。
2. 其余不值得借鉴——**不要为了省事把 PDF 抽完直接塞 LLM 上下文**，那会丢页码、丢引用锚点，正是你这个 RAG 项目要避免的坑。

---

## 三、Claude（Projects / Attachments）

Claude 的引用能力在 **API 层**非常工程化、设计得很干净；消费者端是其 UI 呈现。

### 1. 核心文件 / 论文问答功能
- **【已核验】附件上限**：PDF 最多 1000 页；Project 文件单个 30MB；文件数不限，但总内容必须塞进 Claude 上下文窗口。
  来源：https://support.claude.com/en/articles/8241126-upload-files-to-claude
- **【已核验】支持 PDF（含多模态 PDF 读图）、纯文本；.docx/.xlsx 需先转纯文本**。
  来源：https://platform.claude.com/docs/en/build-with-claude/citations

### 2. 文档管理与交互设计
- **【已核验】Projects**：项目里挂文件 + 自定义上下文，项目内对话复用。
- **【已核验】Artifacts 侧栏**：生成的文档 / 代码 / 应用有专门的 Artifacts 空间统一管理、可整理收藏。
  来源：https://support.claude.com/en/articles/9487310-what-are-artifacts-and-how-do-i-use-them

### 3. 引用溯源设计（最重要）
- **【已核验·API】引用粒度 = 句子级 chunk**：文档先被"chunked"成句子（默认句子切分），最小可引用单元是一句话，也可把连续多句合成一段引用。
  来源：https://platform.claude.com/docs/en/build-with-claude/citations
- **【已核验·API】三种坐标体系**：
  - PDF → **页码区间（1-indexed）**；
  - 纯文本 → **字符下标区间（0-indexed）**；
  - 自定义内容 → **block 下标区间**。
  - 每条引用还带 `document_index`（哪份文档，0-indexed）和 `cited_text`（原话，且不计入输出 token）。
- **【已核验·API】响应结构 = 多 text block 拼接**：每个 text block 内含"一句主张 + 支撑它的引用列表"，流式时以 `citations_delta` 逐个吐出。
- **【已核验】Claude for Word 给出"clickable section citations"**：在 Word 里提问，回答带可点击的章节级引用。
  来源：https://support.claude.com/en/articles/14465370-use-claude-for-word
- **【已核验】联网搜索回答**：每条都带 direct citations、来源链接、相关引文。
  来源：https://support.claude.com/en/articles/10684626-enable-and-use-web-search
- **【推断】消费者网页端把上述引用渲染成行内小角标 / 可点击标记**（API 已验证坐标能力；具体 UI 样式官方未逐屏截图，按其设计惯例推断）。

### 4. 多文档问答与跨文档能力
- **【已核验】单次请求可带多份文档**，`document_index` 区分引用来自哪一份；引用必须"全启用或全不启用"，不能部分文档开引用。

### 5. 架构线索
- **【已核验】引用由模型在生成时按 chunk 边界吐出坐标**（标准化格式输出，再由系统解析成 cited_text + 位置），不是事后匹配。
- **【推断】底层仍是长上下文吞文档 + 句子切分**（引用要能落到字符 / 页坐标，文档全文须在上下文里），与 NotebookLM 的"预建 chunk 索引"路线不同——这是两种 RAG / 长上下文流派的差别。

### 6. 写作 / 笔记辅助能力
- **【已核验】Artifacts** 可直接产出文档 / 网页 / 代码并侧栏管理；Claude for Word 支持修订模式、批注线程。

### 7. 免费 / 付费门槛
- **【推断】免费档有使用额度限制；Pro 约 $20/月解锁更高额度**（官方未在本次检索中给出精确数字，按市面通行口径）。

### 8. 对个人项目的可借鉴程度：**高（工程实现层）**
一句话：**它的"句子级切分 + 三类坐标 + cited_text 原话"是你引用溯源 [n] 后端设计的直接蓝本。**

可直接搬进个人项目的点：
1. **给每个 chunk 存"坐标"**：PDF 存 `(doc_id, page, start_char, end_char)`，纯文本存 `(doc_id, start_char, end_char)`——你的 Chroma 元数据里就该带这些字段，前端点击 [n] 才能跳回去。
2. **引用与主张绑定**：不要把所有引用堆在回答末尾，而是"哪句话 → 哪条引用"就近挂（Claude 的多 text-block 思路）。
3. **`cited_text` 原话冗余存一份**：前端悬停直接显示存好的原话，不必再回原文查。

---

## 四、豆包（国内）

### 1. 核心文件 / 文档问答功能
- **【已核验】"附件或网页问答"属普通免费功能**：快速模式下的普通对话、联网搜索、帮我写作、附件或网页问答、拍图问答均可免费使用。
  来源：豆包权威数据库《豆包专业版》
- **【二手报道】**豆包读 12 页图文 PDF 约 33 秒、抓主旨快，但遇三级交叉表格会数据漏读、格式错乱。
  来源：https://news.sohu.com/a/1082683490_255783

### 2. 文档管理与交互设计
- **【推断】**以会话为单位挂附件，未见 NotebookLM 式的"项目 / 来源库"长期组织；文件随会话走。

### 3. 引用溯源设计
- **【推断】**本次检索未找到豆包附件问答有"行内引用角标 / 点击跳原文"的官方说明；按其产品形态判断，**文档问答偏"读后直接答"，未见系统的 chunk 级引用锚点设计**（无确证，标推断）。

### 4–6. 多文档 / 架构 / 写作辅助
- **【推断】**多文件、联网问答、AI PPT、AI 表格、录音纪要、深入研究等为其高级功能；架构细节不公开。

### 7. 免费 / 付费门槛
- **【已核验】**附件 / 网页问答免费；专业版连续包月 ¥68/月、连续包年 ¥688/年；另有标准 / 加强 / 专业多档按额度分级。
  来源：豆包权威数据库《豆包专业版开通与购买》

### 8. 对个人项目的可借鉴程度：**低**
一句话：**它不做精细引用锚点，对论文问答项目参考价值有限。**
不值得借鉴——其优势在"快、综合功能多"，而非"读论文可溯源"，恰是你项目要差异化的方向。

---

## 五、Kimi（国内，长文档见长）

### 1. 核心文件 / 文档问答功能
- **【已核验】消费者端长文本**：多轮合计支持约 20 万字输入输出；可发 TXT、PDF、Word、PPT、Excel、网址，读后回复。
  来源：https://kimi.moonshot.cn/share/cs50270d4n1ibpv6sjt0
- **【已核验·API】文件问答两步走**：`/v1/files` 上传 → `file-extract` 抽文本 → 把抽好的文本塞进 messages；支持单文件 / 多文件问答；内置 Context Caching 降低反复查同一文档的成本。
  来源：https://platform.kimi.com/docs/guide/use-kimi-api-for-file-based-qa ；https://platform.kimi.ai/docs/guide/ask-questions-about-pdf-content

### 2. 文档管理与交互设计
- **【已核验】Kimi Work（桌面 agent）**：能读本地文件、操作浏览器、跑代码、定时任务，交付完成品；会话内支持 Side Chat（侧聊继承上下文）、选中文本引用。
  来源：https://www.kimi.com/zh-cn/resources/kimi-work-introduction ；https://www.kimi.ai/help/kimi-work/release-notes

### 3. 引用溯源设计
- **【推断】**其路线是"抽全文进长上下文"（API 文档明确把 extract 后的文本整段塞进 messages），**因此消费者端问答偏长上下文直答，未见 NotebookLM / Claude 那种 chunk 级行内引用锚点**。
- **【已核验】**其 Deep Research 产出的报告带"可追溯来源（traceable sources）"，但那是研究报告场景、不是单文档问答。
  来源：https://www.kimi.ai/help/deep-research/deep-research-overview

### 4–6. 多文档 / 架构 / 写作辅助
- **【已核验】**多文件问答靠长上下文；写作 / 翻译 / 长文本生成为强项。

### 7. 免费 / 付费门槛
- **【推断】**消费者端基础长文档问答免费，高级 agent（Kimi Work）/ 深度研究有额度或会员门槛（官方未在本次检索给出精确价）。

### 8. 对个人项目的可借鉴程度：**低–中**
一句话：**它证明了"长上下文吞文档"一派能做，但放弃了引用锚点——你的混合检索 + 溯源路线是另一条路。**
可借鉴点（仅 1 条）：
1. **Context Caching 思路**：反复问同一篇论文时缓存已抽取的文本 / embedding，省时间——你的 Chroma 持久化其实就是这个思路，可在报告里点明。

---

## 六、通义千问 / Qwen（国内）

### 1. 核心文件 / 文档问答功能
- **【已核验】Qwen Chat**：接受 PDF、Word、电子表格、演示文稿、图片，单文件最大 200MB，可一次传多文件并跨文件提问。
  来源：https://qwen3lm.com/chat/
- **【已核验·API】**Qwen-Doc-Turbo / Qwen-Long 同样走"上传 → file-extract 解析 → 返回 file-id → 对话引用"的长上下文路线。
  来源：https://help.aliyun.com/zh/model-studio/data-mining-qwen-doc

### 2–3. 文档管理与引用溯源
- **【二手报道】**媒体实测：通义千问"慢而精"，**自动分章节，关键数据可精准溯源至原文页码**。
  来源：https://news.sohu.com/a/1082683490_255783（2026-09 媒体评测，非官方）
- **【已核验·企业产品】**阿里云"智能问答"有"Reference localization：每条回答标注来源文件的页与段，供核对原文"。
  来源：https://www.alibabacloud.com/help/en/energy-expert/user-guide/intelligent-q-a
  （注意：这是企业行业方案，不是消费者 qwen chat 的承诺。）

### 4–6. 多文档 / 架构 / 写作辅助
- **【推断】**消费端多文件跨答；企业版有精准检索 + 多轮对话 + 引用定位。

### 7. 免费 / 付费门槛
- **【推断】**消费者端基础文档问答免费，企业 API 按 token 计费。

### 8. 对个人项目的可借鉴程度：**中**
一句话：**它是国内产品里少数被评测出"能溯源到页码"的，方向与你一致，但实现不公开。**
可借鉴点（1 条）：
1. **"自动分章节"**：读论文时按章节组织 chunk（而不是固定长度硬切），回答里标注出自哪一章哪一页——对论文这种结构化文档特别有用，可作为你 BM25/向量切分的改进方向。

---

## 七、通用助手类对个人项目的借鉴总结

### 7.1 两类技术路线的分野（最重要的认知）
| 路线 | 代表 | 做法 | 引用锚点 |
|---|---|---|---|
| **长上下文吞文档** | ChatGPT、Kimi、通义、豆包 | 抽全文 → 塞进 LLM 上下文 | 基本不做 / 靠提示词硬逼，页码会丢 |
| **RAG + chunk 级溯源** | NotebookLM、Claude（API） | 预切 chunk → 检索 → 生成时带回坐标 | 行内数字角标，悬停看原话、点击跳原文 |

**你的项目（bge-m3 + Chroma + BM25 + RRF）属于第二类**，这正是与 ChatGPT/Kimi 通用助手差异化的地方——**面试时要明确说："我没有走把 PDF 全塞进上下文的捷径，而是做了混合检索 + 引用坐标，所以能溯源到段落。"**

### 7.2 最该抄的三个交互（按优先级）
1. **NotebookLM 式 [n] 引用三件套**：行内数字角标 → 悬停弹出该 chunk 原话引文 → 点击在右侧原文区滚动并高亮划线。这是把你已有的 [n] 从"纯文本"升级成"可交互溯源"的关键，演示冲击力最大。
2. **Claude 式 chunk 坐标元数据**：在 Chroma 里给每个 chunk 存 `(doc_id, page, start_char, end_char, cited_text)`，前端才能跳回原文；引用就近挂在主张后面，而不是堆在末尾。
3. **NotebookLM 式来源复选框 + Library 式文件库**：上传多篇论文后可勾选本次用哪几篇、文件上传一次多轮复用——低成本、高完成度感。

### 7.3 该警惕的坑
- **引用漂移（citation drift）**：数字角标看着可信，但可能指向"相关却不支撑"的段落。建议在项目里加一步 **Rerank（重排）**或让 LLM 输出时带上原话，降低漂移——这也是你下一步可做的升级点。
- **不要学 ChatGPT 把 PDF 抽完直接塞上下文**：会丢页码、丢锚点，与论文问答的"可溯源"目标相悖。
- **Audio Overview / 播客、Artifacts 侧栏**这类花活对求职项目**优先级低**，先把"上传论文 → 混合检索 → 带 [n] 跳原文"的闭环做扎实。

### 7.4 一句话定位
> 你的「论文阅读问答助手」= NotebookLM 的引用溯源交互 + Claude 的 chunk 坐标工程 + 自研 BM25/RRF 混合检索。把"点击 [n] 高亮原文"做出来，就是这个项目在面试里最能打的差异点。

---

## 附：主要来源清单
- NotebookLM 官方帮助（聊天/引用）：https://support.google.com/notebooklm/answer/16179559?hl=en
- NotebookLM 官方帮助（来源格式）：https://support.google.com/notebooklm/answer/16215270?hl=en
- NotebookLM 官方帮助（笔记导出）：https://support.google.com/notebooklm/answer/16262519?hl=en
- Google for Education NotebookLM 额度：https://edu.google.com/intl/ALL_us/ai-notebooklm/
- OpenAI 帮助（文件库 / 发布说明）：https://help.openai.com/zh-hans-cn/articles/20001052-using-library-to-manage-files-in-chatgpt ；https://help.openai.com/zh-hans-cn/articles/6825453-chatgpt-release-notes
- Claude 官方文档（Citations API）：https://platform.claude.com/docs/en/build-with-claude/citations
- Claude 官方帮助（上传文件 / Word / Web 搜索）：https://support.claude.com/en/articles/8241126-upload-files-to-claude ；https://support.claude.com/en/articles/14465370-use-claude-for-word
- Kimi API 文档（文件问答）：https://platform.kimi.com/docs/guide/use-kimi-api-for-file-based-qa
- 阿里云百炼（Qwen 文件抽取 / 智能问答）：https://help.aliyun.com/zh/model-studio/data-mining-qwen-doc ；https://www.alibabacloud.com/help/en/energy-expert/user-guide/intelligent-q-a
- 豆包权威数据库《豆包专业版》（额度 / 价格，站内资料）
