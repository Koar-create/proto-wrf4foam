---
name: generate-pptx-pptxgenjs
description: 在本仓库内把文字/Markdown内容生成可复现的 pptxgenjs 脚本，并产出 output.pptx。适用于"生成PPT/导出PPTX/做演示文稿"等请求。
---

# 生成 PPTX（pptxgenjs 脚本驱动）

当用户要求"生成PPT / 导出演示文稿 / 输出pptx"时，使用 **Node.js + pptxgenjs** 生成一个可运行的脚本（默认 `generate-slides.js`），在本地生成真实的 `.pptx` 文件（默认 `output.pptx`）。

---

## 目标与约束（必须遵守）

- **输入**：用户提供的 Markdown/纯文字/要点/数据（可包含章节结构与图表意图）。
- **输出（必须）**：
  - `generate-slides.js`：完整可执行的 pptxgenjs 脚本
  - `output.pptx`：脚本运行生成的 PPTX 文件（写死该名字，除非用户指定）
- **脚本规范（必须）**：
  - 使用 `require("pptxgenjs")`（CommonJS）。
  - 不要用 `async/await` 包裹顶层代码，直接同步构建并调用 `pptx.writeFile({ fileName: "output.pptx" })`。
  - 所有坐标与尺寸按 pptxgenjs 的默认单位（**英寸**）。

---

## 第一步（强制）：Markdown 清洗函数

在脚本顶部定义 `clean(text)` 函数，**所有**文本写入 pptxgenjs 前必须先过这个函数，否则 `**`、`*`、`$...$` 等符号会原样出现在幻灯片里：

```js
function clean(text) {
  return text
    // LaTeX 数学块
    .replace(/\$\$[\s\S]*?\$\$/g, '[formula]')
    // LaTeX 行内：先替换常见符号，再去掉 $ 分隔符
    .replace(/\$([^$]+)\$/g, (_, m) => m
      .replace(/\\times/g, '×')
      .replace(/\\cdot/g, '·')
      .replace(/\\frac\{([^}]+)\}\{([^}]+)\}/g, '($1/$2)')
      .replace(/10\^\{([^}]+)\}/g, '10^$1')
      .replace(/\^{([^}]+)}/g, '^$1')
      .replace(/_{([^}]+)}/g, '_$1')
      .replace(/\\[a-zA-Z]+\{([^}]*)\}/g, '$1')
      .replace(/\\/g, '')
    )
    // Markdown 格式标记
    .replace(/\*\*([^*]+)\*\*/g, '$1')   // **bold**
    .replace(/\*([^*]+)\*/g, '$1')        // *italic*
    .replace(/`([^`]+)`/g, '$1')          // `code`
    // 残留符号
    .replace(/[*_`~]/g, '')
    .trim();
}
```

---

## 第二步（强制）：版式系统

pptxgenjs 没有模板继承，所有版式靠**封装函数**实现一致性。脚本中必须定义以下常量和函数：

### 配色与字体常量

```js
const C = {
  navy:    '1E2761',   // 深海军蓝 —— 封面/章节页背景、标题栏
  navyDk:  '24307A',   // 稍深蓝 —— 封面底部信息块
  accent:  '5CC8FF',   // 亮蓝色 —— 强调线、页码框、装饰点
  white:   'FFFFFF',
  offWhite:'F8FAFC',   // 内容页左栏浅底
  textDk:  '111827',   // 正文深灰
  textMid: '64748B',   // 副标题、图注
  textLt:  'DCE6FF',   // 封面/深色页副文本
};

const F = {
  title:  'Calibri',
  body:   'Calibri',
};
```

### 版式函数

#### addCoverSlide(pptx, title, subtitle, authors, journal)
封面页，深色背景：

```js
function addCoverSlide(pptx, title, subtitle, authors, journal) {
  const slide = pptx.addSlide();
  // 全页深色背景
  slide.addShape(pptx.ShapeType.rect, { x:0, y:0, w:'100%', h:'100%', fill:{color:C.navy} });
  // 顶部装饰线（1px 亮蓝）
  slide.addShape(pptx.ShapeType.rect, { x:0, y:0, w:'100%', h:0.06, fill:{color:C.accent} });
  // 主标题
  slide.addText(clean(title), {
    x:0.9, y:1.5, w:11.6, h:1.0,
    fontSize:48, bold:true, color:C.white, fontFace:F.title,
  });
  // 副标题
  if (subtitle) slide.addText(clean(subtitle), {
    x:0.9, y:2.55, w:11.6, h:0.9,
    fontSize:22, color:C.textLt, fontFace:F.body,
  });
  // 底部信息块
  slide.addShape(pptx.ShapeType.rect, { x:0.9, y:4.4, w:11.6, h:1.7, fill:{color:C.navyDk} });
  const info = [authors, journal].filter(Boolean).join('\n');
  if (info) slide.addText(clean(info), {
    x:1.2, y:4.65, w:11.0, h:1.2,
    fontSize:16, color:C.white, fontFace:F.body,
  });
  return slide;
}
```

#### addSectionSlide(pptx, partLabel, sectionTitle)
章节分隔页（深色）：

```js
function addSectionSlide(pptx, partLabel, sectionTitle) {
  const slide = pptx.addSlide();
  slide.addShape(pptx.ShapeType.rect, { x:0, y:0, w:'100%', h:'100%', fill:{color:C.navy} });
  slide.addShape(pptx.ShapeType.rect, { x:0, y:5.9, w:'100%', h:1.6, fill:{color:C.navyDk} });
  slide.addShape(pptx.ShapeType.rect, { x:0, y:0, w:'100%', h:0.06, fill:{color:C.accent} });
  slide.addText(clean(sectionTitle), {
    x:0.9, y:2.6, w:11.6, h:0.9,
    fontSize:54, bold:true, color:C.white, fontFace:F.title,
  });
  if (partLabel) slide.addText(clean(partLabel), {
    x:0.9, y:3.6, w:11.6, h:0.6,
    fontSize:20, color:C.textLt, fontFace:F.body,
  });
  return slide;
}
```

#### addContentHeader(slide, pptx, title, pageNum)
内容页公共标题栏（所有内容页调用此函数添加顶栏）：

```js
function addContentHeader(slide, pptx, title, pageNum) {
  // 深色标题栏
  slide.addShape(pptx.ShapeType.rect, { x:0, y:0, w:'100%', h:0.75, fill:{color:C.navy} });
  // 标题文字
  slide.addText(clean(title), {
    x:0.65, y:0.15, w:11.6, h:0.5,
    fontSize:28, bold:true, color:C.white, fontFace:F.title,
  });
  // 页码框（亮蓝色小方块）
  slide.addShape(pptx.ShapeType.rect, { x:12.70, y:0.18, w:0.55, h:0.40, fill:{color:C.accent} });
  slide.addText(String(pageNum), {
    x:12.72, y:0.20, w:0.51, h:0.36,
    fontSize:16, bold:true, color:C.navy, align:'center', fontFace:F.body,
  });
}
```

---

## 第三步（强制）：内容页版式选择

根据幻灯片内容从以下三种版式中选择，**同一演示文稿内应混用这三种版式，避免全篇一律**：

### Layout A：文字+图片（左文右图）

适用于：有配图的内容页（最常见）。

```
┌─────────────────────────────────────────────┐
│  ■ 标题栏（全宽深色，0.75"高）            │
├──────────────┬──────────────────────────────┤
│              │  [图片标题]                  │
│  要点列表    │  ──────────────────          │
│  (左栏，     │                              │
│   浅灰底)    │  [图片区域]                  │
│              │                              │
│              │  图注（小字）                │
└──────────────┴──────────────────────────────┘
```

```js
// 左栏：浅底+要点
slide.addShape(pptx.ShapeType.rect, { x:0.65, y:1.15, w:6.00, h:5.75, fill:{color:C.offWhite} });
slide.addText(bullets, { x:1.0, y:1.5, w:5.3, h:5.05, fontSize:16, color:C.textDk, ... });

// 右栏：图片+图注
slide.addShape(pptx.ShapeType.rect, { x:7.0, y:1.15, w:6.0, h:5.75, fill:{color:C.white} });
slide.addShape(pptx.ShapeType.rect, { x:7.30, y:1.80, w:5.40, h:0.12, fill:{color:C.accent} }); // 亮蓝装饰线
slide.addText(figTitle, { x:7.35, y:1.43, w:5.3, h:0.35, fontSize:14, bold:true, color:C.textMid });
slide.addImage({ path: imgPath, x:7.35, y:1.85, w:5.3, h:4.6 });
slide.addText(caption, { x:7.35, y:6.65, w:5.3, h:0.25, fontSize:10, color:C.textMid });
```

### Layout B：纯文字分栏（左主右高亮）

适用于：无图、有对比概念的内容页。左栏放主要内容，右栏放关键要点或对比信息。

```
┌─────────────────────────────────────────────┐
│  ■ 标题栏                                  │
├──────────────┬──────────────────────────────┤
│              │  ■ KEY POINTS               │
│  主要内容    │  ──────────────────          │
│  要点列表    │                              │
│  (浅底)      │  高亮摘要                    │
│              │  (白底，右栏)               │
└──────────────┴──────────────────────────────┘
```

```js
// 左栏
slide.addShape(pptx.ShapeType.rect, { x:0.65, y:1.15, w:6.00, h:5.75, fill:{color:C.offWhite} });
slide.addText(mainBullets, { x:1.0, y:1.5, w:5.3, h:5.05, fontSize:16, color:C.textDk, ... });

// 右栏：高亮区
slide.addShape(pptx.ShapeType.rect, { x:7.0, y:1.15, w:6.0, h:5.75, fill:{color:C.white} });
slide.addShape(pptx.ShapeType.rect, { x:7.30, y:1.80, w:5.40, h:0.12, fill:{color:C.accent} });
slide.addText('KEY POINTS', { x:7.35, y:1.43, w:5.3, h:0.35, fontSize:14, bold:true, color:C.textMid });
slide.addText(highlights, { x:7.35, y:2.05, w:5.3, h:4.65, fontSize:16, color:C.textDk, ... });
```

### Layout C：全宽数据页（表格/统计）

适用于：数据表格、统计对比、流程步骤。整个内容区全宽展开，不分栏。

```
┌─────────────────────────────────────────────┐
│  ■ 标题栏                                  │
├─────────────────────────────────────────────┤
│  说明文字（全宽，浅底）                      │
│                                             │
│  ┌──────┬──────┬──────┐                   │
│  │表头  │列1   │列2   │                   │
│  ├──────┼──────┼──────┤                   │
│  │行1   │值    │值    │                   │
│  └──────┴──────┴──────┘                   │
└─────────────────────────────────────────────┘
```

用 `slide.addTable()` 实现，表头行用 `fill: {color: C.navy}` + `color: C.white`，数据行用交替底色。

---

## 第四步（强制）：要点文字的 pptxgenjs 格式

pptxgenjs 支持 **富文本数组**格式，可以在同一个文本框内实现加粗/颜色差异，**比纯文字更接近 Markdown 的原意**：

```js
// 将 Markdown 要点转换为 pptxgenjs 富文本数组
function mdBulletToRichText(lines) {
  return lines.map(line => {
    // 去掉前导 bullet 符号
    const text = line.replace(/^[\s*\-•]+/, '');
    // 识别 **label:** 模式：前半加粗+深色，后半普通
    const boldMatch = text.match(/^([^:]+:)\s*(.*)/);
    if (boldMatch) {
      return [
        { text: '• ' + boldMatch[1] + ' ', options: { bold: true, color: C.textDk, fontSize: 16 } },
        { text: boldMatch[2], options: { bold: false, color: C.textDk, fontSize: 16 } },
        { text: '\n', options: { fontSize: 16 } },
      ];
    }
    return [
      { text: '• ' + clean(text) + '\n', options: { color: C.textDk, fontSize: 16 } },
    ];
  }).flat();
}

// 使用方式：
slide.addText(mdBulletToRichText(lines), {
  x:1.0, y:1.5, w:5.3, h:5.05,
  fontFace: F.body,
  paraSpaceAfter: 8,   // 段后间距（pt）
});
```

> 注意：`paraSpaceAfter` 需要在每段的 options 里设置，或作为文本框全局属性。
> 富文本数组中，段落分隔用 `\n`，每个 `{ text, options }` 对象是一个 run。

---

## 第五步（强制）：图片引用规则

1. **路径必须是调用脚本时的相对路径或绝对路径**，pptxgenjs 在 `writeFile` 时才会读取图片。
2. 图片如果不存在，脚本会抛错。因此：
   - 如果 md 里有 `assets/images/figureN.png` 且该文件确实存在，**直接引用**。
   - 如果图片文件不存在，**用占位矩形 + 说明文字**代替，不要留下损坏引用：
     ```js
     // 占位符函数
     function addImagePlaceholder(slide, x, y, w, h, label) {
       slide.addShape(pptx.ShapeType.rect, { x, y, w, h, fill:{color:'E2E8F0'}, line:{color:'CBD5E1'} });
       slide.addText('[图片: ' + label + ']', {
         x, y, w, h, align:'center', valign:'middle',
         fontSize:14, color:C.textMid, fontFace:F.body,
       });
     }
     ```
3. **图片尺寸控制**：右栏图片区标准尺寸为 `w:5.3, h:4.6`（英寸）。图片会自动填充，不会保持原始比例，如果原图比例差异很大需要调整 h。
4. **全宽图片**（Layout C 数据页偶尔用）：`x:0.65, y:1.1, w:12.0, h:5.8`。

---

## 第六步（强制）：总结页和 Q&A 页

### 总结页
```js
function addSummarySlide(pptx, takeaways, pageNum) {
  const slide = pptx.addSlide();
  addContentHeader(slide, pptx, 'Summary & Conclusions', pageNum);
  slide.addShape(pptx.ShapeType.rect, { x:0.65, y:1.15, w:12.0, h:5.75, fill:{color:C.offWhite} });
  const richText = takeaways.map((t, i) => [
    { text: `${i+1}.  `, options: { bold:true, color:C.accent, fontSize:18 } },
    { text: clean(t) + '\n\n', options: { color:C.textDk, fontSize:16 } },
  ]).flat();
  slide.addText(richText, { x:1.0, y:1.5, w:11.2, h:5.2, fontFace:F.body });
}
```

### Q&A 页（同封面风格，深色）
```js
function addQASlide(pptx, repoUrl) {
  const slide = pptx.addSlide();
  slide.addShape(pptx.ShapeType.rect, { x:0, y:0, w:'100%', h:'100%', fill:{color:C.navy} });
  slide.addShape(pptx.ShapeType.rect, { x:0, y:0, w:'100%', h:0.06, fill:{color:C.accent} });
  slide.addText('Thank You', { x:0.9, y:2.0, w:11.6, h:1.0, fontSize:54, bold:true, color:C.white, fontFace:F.title });
  slide.addText('Questions?', { x:0.9, y:3.1, w:11.6, h:0.7, fontSize:28, color:C.textLt, fontFace:F.body });
  if (repoUrl) slide.addText(repoUrl, { x:0.9, y:5.5, w:11.6, h:0.5, fontSize:16, color:C.accent, fontFace:F.body });
}
```

---

## 生成脚本时的完整工作步骤

1. **解析 Markdown 结构**：提取封面信息（标题/作者/期刊）、章节（`# PART`）、幻灯片（`## Slide`）、图片引用（`figure*.png`）、表格。

2. **为每张幻灯片选择版式**：
   - 有图片引用 → Layout A（文字+图片）
   - 有对比/两列内容 → Layout B（左主右高亮）
   - 有表格/数据 → Layout C（全宽）
   - 纯文字 → Layout B（右栏放关键句，不得留空）

3. **转换要点文字**：调用 `mdBulletToRichText()` 或手动构建富文本数组，保留 `**Label:**` 的视觉区分。

4. **处理图片**：检查 assets 目录，存在则引用，不存在则用 `addImagePlaceholder()`。

5. **整体检查**：
   - 所有文本已过 `clean()` 函数（无残留 `*`、`$`、`\`）
   - 所有内容页都有 `addContentHeader()` 标题栏
   - 没有纯白空白内容区

6. **输出**：告知用户运行 `node generate-slides.js`，产物为 `output.pptx`。

---

## 环境准备

```bash
# 项目内安装（推荐，可复现）
npm init -y
npm i pptxgenjs

# 或全局安装
npm install -g pptxgenjs
```

---

## 常见错误与避免方式

| 错误现象 | 原因 | 解决方式 |
|----------|------|----------|
| 幻灯片出现 `**bold**` 原文 | 忘记调用 `clean()` | 所有文本写入前过 `clean()` |
| 幻灯片出现 `$10^{4}$` 原文 | `clean()` 里 LaTeX 转换漏写 | 检查 `clean()` 的正则覆盖 |
| 右栏图片空白 | 图片路径不存在 | 用 `addImagePlaceholder()` 代替 |
| 全篇同一版式 | 没有混用 Layout A/B/C | 有图用A，对比用B，数据用C |
| 要点没有粗体区分 | 用了纯文字而非富文本数组 | 用 `mdBulletToRichText()` |
| 文字溢出文本框 | 字数过多 / h 太小 | 减少要点（≤6条）或拆为两页 |
| 图片压缩变形 | w/h 比例与原图差异大 | 根据图片原始比例调整 h |