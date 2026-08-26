# ICLR 2027 submission version

这是从 AAAI 版本转换得到的自包含 ICLR 2027 初投稿目录。原 AAAI 目录未修改。

## 主要文件

- `main_iclr_submission.tex`：主入口，含 9 页正文、参考文献入口和附录入口。
- `appendix_iclr.tex`：完整单栏附录。
- `references.bib`：参考文献数据库。
- `iclr2027_conference.sty`、`iclr2027_conference.bst`：官方 2027 样式文件。
- `figures/`、`selected_figure_exports/`：本论文用到的全部图像，路径均为目录内相对路径。
- `output/pdf/main_iclr_2027_submission.pdf`：最终提交优化版。
- `supplementary_material/submission_code_minimal_anonymized.zip`：匿名代码 supplement。
- `ICLR_2027_REQUIREMENTS_AND_COMPLIANCE.md`：要求来源、合规状态和提交前人工检查清单。

## 编译

在本目录执行：

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main_iclr_submission.tex
```

该命令生成未二次压缩的 `main_iclr_submission.pdf`。最终 OpenReview 提交文件为 `output/pdf/main_iclr_2027_submission.pdf`；它使用 300 dpi 彩色/灰度图像重采样来降低体积，文字和矢量内容仍保持清晰。

## 当前验证结果

- 核心正文（含 Conclusion）：第 1–9 页；AI Use Statement、Reproducibility Statement 与参考文献从第 10 页开始，参考文献至第 12 页；附录：第 13–86 页。
- 最终 PDF：86 页，US Letter，24,143,026 bytes。
- 所有字体嵌入；无 overfull box、未定义引用/文献或缺失图片。
- 已渲染并视觉检查全部页面。
