/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_BACKEND_BASE?: string;
  readonly VITE_BACKEND_WS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
