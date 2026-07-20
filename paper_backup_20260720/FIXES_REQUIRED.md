# AAAI-27 投稿合规待改清单(最终核对版)

> 经三轮分析 + 一次第三方裁决 + 逐条在本地源文件/PDF/`aaai2027.sty` 实测后定稿。
> 每条:**文件 → 行号 → 当前 → 改成**。原文件已备份在 `_backup_20260720_155922\`。
> 编译从父目录 `D:\research\UIUC research\paper` 执行(图片在 `figures\`)。

---

## 一、官方依据(实测)

1. AAAI-27 Submission Instructions:US Letter、正文≤7页、全文≤9页(8-9页仅参考文献)、匿名、Type1/TrueType 字体全嵌入、PDF。
2. AAAI Press author guide:**无页码、禁 `\pagestyle`、禁 `\resizebox` 缩表、表 caption 在表下方、caption 用正常 Roman 不加粗、禁负 `\vspace`、禁改参考文献排版**。
3. 本地 `aaai2027.sty` 第358-422行禁用包清单(实测逐字):hyperref/bbm/authblk/balance/CJK/flushend/fullpage/geometry/navigator/savetrees/setspace/stfloats/tabu/titlesec/tocbibind/ulem/wrapfig。**无 multicol**。sty 第75行已 `\RequirePackage{placeins}`。
4. sty 第322-323行实测:`\small` 和 `\footnotesize` **都定义为 9pt**(不是8pt),均属 AAAI 允许的表格字号范围。

---

## 二、主论文 `main_aaai_submission.tex`

### ❌ 1. 删页码(最关键,desk reject 级)
- 行65-66
- 当前:`\thispagestyle{plain}` / `\pagestyle{plain}`
- 改成:删除这两行(sty 第84行默认 `\pagestyle{empty}`,删后无页码)

### ❌ 2. 删参考文献手工挤版整段
- 行915-937
- 改成:
  ```latex
  \bibliography{references}

  \end{document}
  ```
- 原因:`\clearpage`(915)、`\onecolumn`(916)、`\bibfont{9.4pt}`(920)、`\thebibliography` 重定义(922)、`\bibsep{0pt}`(924)、`\vspace{-0.4em}`(934)、`\begin{multicols}{2}`(932)——全部属禁。sty 自动处理参考文献标题/字体/双栏。`multicol` 包本身可留,但这段 multicols 环境连同重定义要删。

### ❌ 3. 删 `\resizebox` 整体缩表(官方点名禁止)
- 行533
- 当前:`\resizebox{\columnwidth}{!}{%` ... `\begin{tabular}{@{}lll@{}}`
- 改成:去掉 `\resizebox{...}{!}{` 和对应闭合 `}`;把 `\begin{table}[t]` 改 `\begin{table*}[t]`(跨双栏),保留 `\footnotesize`(9pt,合规)。
  ```latex
  \centering
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{@{}lll@{}}
  ...
  \end{tabular}
  \end{table*}
  ```
- **注意:`\footnotesize` 是9pt合规,不用改字号**(之前误判过,已纠正)。真正要删的只有 resizebox。

### ❌ 4. 删 `\renewcommand{\arraystretch}` 压行高
- 行657:`\renewcommand{\arraystretch}{0.95}`
- 行831:`\renewcommand{\arraystretch}{0.96}`
- 改成:删除这两行。

### ❌ 5. 删 caption 里的 `\textbf{...}`
- 行659:`\caption{\textbf{Attack Objective Comparison.} ...}`
- 行832:`\caption{\textbf{Adversarial trained models robustness.} ...}`
- 改成:去掉 `\textbf{}`,`\caption{Attack Objective Comparison. ...}`

### ❌ 6. 表 caption 移到表下方
- 行653-661块、行826-837块
- 当前:`\caption{...}\label{...}` 在 `\begin{tabular}` 之前
- 改成:移到 `\end{tabular}` 之后、`\end{table}` 之前。

### ❌ 7. 删图内负行间距
- 行646、647
- 当前:`...png}\\[-0.35em]`
- 改成:`...png}\\`

### ⚠️ 8. 删 `\captionsetup{skip=1pt}`
- 行811、829
- 改成:删除(sty 对此不报错,属通用规则建议)。

### ✅ 9. 浮动体参数(行18-30)——保留,不删
- 当前:`\setcounter{topnumber}{5}` 等 + `\setlength{\textfloatsep}{6pt...}` 等
- **结论:保留**。理由:①sty 不报错(不在禁用清单);②author guide 对此偏"建议";③删除可能导致图乱跑、正文超7页(超页比保留参数严重得多);④实际投稿大量论文保留此类参数控版。
- **如确实想清理**:先备份,删后重编,确认仍≤9页且正文≤7页再保留删除;否则恢复。

---

## 三、Supplement:`main_aaai_submission_supplement.tex`

### ❌ 1. 删页码
- 行55-56 → 删除 `\thispagestyle{plain}` / `\pagestyle{plain}`

### ❌ 2. 删 arXiv keep-alive 死代码块
- 行3106的 `\begingroup` → 对应 `\endgroup`(从 `% arXiv file scanner keep-alive block` 到其 `\endgroup`)
- 改成:整块删除。
- 性质:编译依赖风险(任一图片路径失效即编译失败),非 AAAI 明文格式违规,但删之无保留价值。

### ❌ 3. 修正超宽图
- 行2529、2546、2555、2564:`width=1.02\textwidth` → `width=\textwidth`
- 行2608:`width=1.03\textwidth` → `width=\textwidth`

### ❌ 4. 删参考文献手工挤版整段
- 行3157附近 `\bibfont{9.4pt}` / `\thebibliography` 重定义 / `\bibsep{0pt}` / 行3171 `\vspace{-0.4em}`
- 改成:简化为 `\bibliography{references}` + `\end{document}`。

### ⚠️ 5. 修正硬编码表号
- 行691:`Table~6` → `Table~\ref{tab:attack_step_timing_summary}`
- 行692:`Table~7` → `Table~\ref{tab:backend_bridge_timing}`
- 行693:`Table~8` → `Table~\ref{tab:standalone_component_timing_memory}`

### ✅ 6. `\FloatBarrier`(7处)— 保留
- sty 第75行已 `\RequirePackage{placeins}`,已编译验证无报错。Supplement 用它控图分页,保留合理。

### ✅ 7. `\clearpage`(用于整页大图大表)— 保留
- Supplement 官方无逐项禁令;用于分隔整页大图大表属合理。

### ⚠️ 8. 清理 `\captionsetup{font=...}` 字号类
- supplement 里多处 `\captionsetup{font=small/footnotesize,skip=1pt}`:删 `font=` 参数(会把 caption 改成9pt),`skip=` 可删可不删。属低风险。

---

## 四、Supplement:`main_aaai_submission_supplement_under10mb.tex`

(与主 supplement 结构一致,行号偏移)

### ❌ 1. 删页码 — 行57-58
### ❌ 2. 删 arXiv 死块 — 行3077的 `\begingroup` 到 `\endgroup`
### ❌ 3. 修正超宽图 — 行2521/2538/2547/2556(`1.02`)、行2600(`1.03`)→ `width=\textwidth`
### ❌ 4. 删参考文献挤版 — 行3125附近 + 行3139 `\vspace{-0.4em}`
### ⚠️ 5. 修正硬编码表号 — 行693/694/695 → `\ref{...}`
### ✅ 6. `\FloatBarrier` — 保留
### ✅ 7. `\clearpage` — 保留
### ⚠️ 8. 清理 `\captionsetup{font=}` — 同上

---

## 五、🚫 不要改(误判纠正)

- **不要加 `\usepackage{times}`** — sty 第67-68行明确禁;已自动加载 newtxtext/helvet/courier。
- **不要删 `\usepackage{multicol}`** — 不在 sty 禁用清单。要删的只是参考文献那段对 `thebibliography` 的重定义 multicols 环境,不是包。
- **不要删 `\FloatBarrier`** — placeins 已由 sty 加载。
- **`\footnotesize` 是9pt合规,不用改字号** — 只删 `\resizebox`。(此前误判,已纠正)
- **`\author{Anonymous Authors}` 不用改** — 不破匿名。
- **`\frenchspacing` 不强制** — 加不加均可,sty 不检测。

---

## 六、改完验证(在 `D:\research\UIUC research\paper\` 下)

```powershell
# 1. 重编
pdflatex -interaction=nonstopmode main_aaai_submission_version\main_aaai_submission.tex
pdflatex -interaction=nonstopmode main_aaai_submission_version\main_aaai_submission_supplement.tex
pdflatex -interaction=nonstopmode main_aaai_submission_version\main_aaai_submission_supplement_under10mb.tex
# 2. 查页码(应无数字)
pdftotext -f 1 -l 8 -layout main_aaai_submission_version\main_aaai_submission.pdf - | Select-Object -Last 3
# 3. 查页数/纸张
pdfinfo main_aaai_submission_version\main_aaai_submission.pdf | Select-String "Pages|Page size"
# 4. 查字体(主论文应无 Type 3)
pdffonts main_aaai_submission_version\main_aaai_submission.pdf | Select-String "Type 3"
# 5. 确认正文≤7页,8-9页仅参考文献
```

---

## 七、改动顺序

1. 主论文1(页码)→ 立即重编验证
2. 主论文2 + 两supplement 4(参考文献挤版)
3. 主论文3(resizebox)
4. 主论文4/5/6/7(arraystretch/textbf/caption位置/负间距)
5. 两supplement 2(arXiv死块)
6. 两supplement 3(超宽图)
7. 两supplement 5(硬编码表号)
8. 低风险项(主论文8、supplement 8)
9. **主论文9(浮动参数)默认不动**;仅在你愿意时备份后试删+重编确认不超页

每改一组重编一次,确认无新报错且页数合规,再改下一组。
