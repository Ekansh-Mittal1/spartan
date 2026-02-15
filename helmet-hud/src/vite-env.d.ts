/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_WS_URL: string;
  readonly VITE_STREAM_BASE: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
