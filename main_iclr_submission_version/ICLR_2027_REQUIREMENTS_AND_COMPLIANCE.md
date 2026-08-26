# ICLR 2027 投稿要求与本版本合规记录

核查日期：2026-08-26（America/Chicago）。本记录以 ICLR 2027 官方网页、官方 LaTeX 样式包和 OpenReview 投稿表单为准。

## 官方来源

- [ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
- [ICLR 2027 官方 LaTeX 样式包](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip)
- [ICLR 2027 AI Policy for Authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors)
- [ICLR 2027 Call for Papers](https://www.iclr.cc/Conferences/2027/CallForPapers)
- [ICLR 2027 OpenReview 投稿表单定义](https://api2.openreview.net/invitations?id=ICLR.cc%2F2027%2FConference%2F-%2FSubmission)

## 主要硬性要求

| 项目 | ICLR 2027 官方要求 | 本版本 |
|---|---|---|
| 初投稿正文 | 最多 9 页；严格执行 | 正文在第 9 页结束，符合 |
| rebuttal / camera-ready 正文 | 最多 10 页 | 本文件按初投稿 9 页准备 |
| 参考文献 | 不计入正文页数，不限页数 | 第 10 页开始，第 12 页结束；第 10 页页首为 AI 与可复现性声明 |
| 附录 | 放在参考文献之后，不限页数；审稿人没有义务阅读 | 第 13–86 页，共 74 页 |
| 正文、参考文献、附录关系 | 官方鼓励论文与补充文字合并为一个 PDF；也允许另交匿名 supplement | 合并为一个 PDF，顺序为正文→参考文献→附录 |
| 总页数 | 官方没有另设总页数上限 | 86 页；受 50 MB 文件上限约束 |
| 字数 | 官方没有另设总字数上限 | 以 9 页正文上限控制 |
| 版式 | 官方 ICLR 2027 样式，单栏，US Letter | 使用原版 `iclr2027_conference.sty`，单栏，612×792 pt |
| 版心 | 5.5 in × 9.0 in；左右各 1.5 in，垂直布局由官方样式控制 | 未覆盖任何 margin、textwidth、textheight、topmargin 参数 |
| 正文字体 | 10 pt Times 系，11 pt 行距 | 使用 `times` 与官方 `\normalsize` |
| 表格文字 | 官方没有另给表格缩字号许可；应遵守通用字号和可读性 | 所有遗留 `\footnotesize` 在附录中映射为 10 pt `\normalsize`；没有 `\tiny` / `\scriptsize` |
| 标题与小标题 | 由官方样式控制；标题约 17 pt 小型大写，一级标题 12 pt 小型大写，二/三级标题 10 pt | 未改写官方标题和标题层级字号 |
| 页码和行号 | 投稿模式由官方样式生成 | 保留官方页眉、页码和灰色行号 |
| 匿名 | 双盲；主文及补充材料不得暴露作者身份；自引应使用第三人称 | 作者显示为 `Anonymous Authors`；PDF Author 元数据为空；代码包为匿名版 |
| AI 使用声明 | 必须披露生成式 AI 的使用；作者对内容负责 | 第 10 页、参考文献之前含 `AI Use Statement` |
| 可复现性声明 | 官方建议在主文中加入简短说明，不计正文页数 | 第 10 页、参考文献之前含 `Reproducibility Statement` |
| Ethics Statement | 可选；如包含，最多 1 页且不计正文页数 | 当前未单列；作者应按研究实际决定是否加入 |
| 主 PDF 大小 | OpenReview 表单 `maxSize: 50`，即最多 50 MB | 提交版 24,143,026 bytes（约 24.14 MB） |
| supplement | 可选、必须匿名；OpenReview 接受 `.pdf` 或 `.zip`，最大 100 MB | 匿名代码 ZIP 为 345,787 bytes |

## 图、表与 PDF 注意事项

- 图必须清楚、深色、可辨认并连续编号；图题放在图下。表应整洁、居中、连续编号；表题放在表上。不要通过缩小字体或超出版心来塞内容。
- 线图/流程图宜使用 PDF 矢量图；位图宜使用足够分辨率的 PNG/JPEG。使用 `pdflatex` 时不要直接依赖 EPS。
- 不得修改官方样式的页面尺寸、版心、字号和标题格式。本版本没有使用 `geometry`、`\fontsize`、负边距、双栏切换或微型字体。
- 主 PDF 与 supplement 都必须匿名；不要在文件名、PDF 元数据、致谢、代码路径、仓库链接、图片水印或附录中泄露身份。
- 投稿期间不得同时在其他会议处于审稿状态；还应遵守抄袭、重复投稿、审稿分配、作者列表冻结和 OpenReview 账户要求。最终提交前应重新阅读官方 Author Guidelines，因为会议方可能更新操作细节。

## 本版本的处理方式

- 从 `main_aaai_submission_version` 复制论文所需的 LaTeX、BibTeX 和全部被引用图像，源目录未修改。
- 采用官方 ICLR 2027 `.sty` 和 `.bst`，将 AAAI 双栏排版转换为 ICLR 单栏。
- 在不缩小官方字号、不改版心的前提下，将核心正文控制在 9 页；原“Main-Paper Auxiliary Figures”中的 Figure 2–8 已全部移回正文，并通过并排组合和图像缩放保留完整内容；技术附录和其余实验图表继续保留。
- 将原补充材料中为 AAAI 双栏而使用的跨栏命令移除，并拆分过宽表格，使其适配 5.5 in 单栏版心。
- 最终 LaTeX 日志中没有 overfull box、未定义引用/文献、重复 PDF 锚点或缺失文件警告。
- 最终 PDF 的全部字体均嵌入，页面为 US Letter，无加密、无 JavaScript、Author 元数据为空。
- 将全部 86 页以 PNG 方式进行版面核查，并重点放大检查 Figure 1–8 所在的第 6–9 页、声明与参考文献过渡页、长表页、流程图页、密集位图页和末页。未发现裁切、超出版心、重叠或压缩导致的明显失真。
- 已解包扫描匿名代码 supplement 的 60 个文件；文件名和文本中未发现姓名、UIUC/Illinois 标识、邮箱、用户目录绝对路径、个人主页或外部仓库 URL。

## 作者提交前必须人工确认

1. `AI Use Statement` 目前写明 AI 用于排版和语言编辑。请按真实使用情况补全或修改；不要仅因模板合规而保留不真实的声明。
2. `Reproducibility Statement` 写明匿名代码包随投稿提供。上传主 PDF 时应同时上传 `supplementary_material/submission_code_minimal_anonymized.zip`，否则需改写该句。
3. 当前自动扫描未发现身份线索；上传前仍建议人工确认 ZIP 中没有自动扫描无法识别的用户名、Git 历史、日志、模型平台账户或非匿名 URL。
4. 在 OpenReview 锁定作者列表前确认所有作者的姓名、邮箱、机构、个人主页和冲突域信息；这些信息填在 OpenReview，不写入匿名 PDF。
5. 在截止日前确认官方页面没有发布新版样式或勘误。ICLR 2027 摘要截止为 2026-09-18 23:59 AoE，全文截止为 2026-09-25 23:59 AoE。
