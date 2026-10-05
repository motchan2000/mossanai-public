// Minecraft のスクリプト実行環境で使えるグローバル（ES標準ライブラリ以外）
declare const console: {
  log(...data: unknown[]): void;
  info(...data: unknown[]): void;
  warn(...data: unknown[]): void;
  error(...data: unknown[]): void;
};
