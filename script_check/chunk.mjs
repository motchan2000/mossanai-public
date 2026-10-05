// JS/TS ファイルをトップレベルの宣言単位に分割して JSON で出力する（TypeScript の構文解析器を使用）。
//   node chunk.mjs <file> [<file> ...]
// 出力: [{ file, imports: "...", units: [{ kind, name, start, end, text }] }]
import fs from "node:fs";
import ts from "typescript";

const out = [];
for (const file of process.argv.slice(2)) {
  const text = fs.readFileSync(file, "utf8");
  const sf = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true,
    file.endsWith(".ts") ? ts.ScriptKind.TS : ts.ScriptKind.JS);
  const imports = [];
  const units = [];
  for (const st of sf.statements) {
    const start = st.getFullStart();
    const body = text.slice(start, st.getEnd()).replace(/^\s*\n/, "");
    if (ts.isImportDeclaration(st)) {
      imports.push(st.getText(sf));
      continue;
    }
    let kind = ts.SyntaxKind[st.kind];
    let name = "";
    if (ts.isFunctionDeclaration(st) || ts.isClassDeclaration(st)) {
      name = st.name ? st.name.text : "";
    } else if (ts.isVariableStatement(st)) {
      name = st.declarationList.declarations.map((d) => d.name.getText(sf)).join(",");
    } else if (ts.isExpressionStatement(st)) {
      // world.afterEvents.xxx.subscribe(...) など
      name = st.expression.getText(sf).split("(")[0].slice(0, 80);
    }
    const line = (pos) => sf.getLineAndCharacterOfPosition(pos).line + 1;
    units.push({ kind, name, start: line(st.getStart(sf)), end: line(st.getEnd()), text: body });
  }
  out.push({ file, imports: imports.join("\n"), units });
}
process.stdout.write(JSON.stringify(out));
