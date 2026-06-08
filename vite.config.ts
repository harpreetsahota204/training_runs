import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { externalizeDeps } from "vite-plugin-externalize-deps";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const EXTERNALS = [
  "@fiftyone/components",
  "@fiftyone/operators",
  "@fiftyone/state",
  "@fiftyone/utilities",
  "@fiftyone/spaces",
  "@fiftyone/plugins",
  "@fiftyone/aggregations",
  "@fiftyone/core",
  "styled-components",
  "recoil",
  "react",
  "react-dom",
  "@mui/material",
];

export default defineConfig({
  mode: "development",
  plugins: [
    react({ jsxRuntime: "classic" }),
    externalizeDeps({
      deps: false,
      devDeps: false,
      useFile: join(__dirname, "package.json"),
      include: EXTERNALS,
    }),
  ],
  build: {
    minify: true,
    lib: {
      entry: join(__dirname, "src/index.tsx"),
      name: "@voxel51/training-runs",
      fileName: (format) => `index.${format}.js`,
      formats: ["umd"],
    },
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: true,
    rollupOptions: {
      output: {
        globals: {
          react: "React",
          "react-dom": "ReactDOM",
          recoil: "recoil",
          "jsx-runtime": "jsx",
          "react/jsx-runtime": "jsx",
          "styled-components": "__styled__",
          "@fiftyone/plugins": "__fop__",
          "@fiftyone/operators": "__foo__",
          "@fiftyone/state": "__fos__",
          "@fiftyone/components": "__foc__",
          "@fiftyone/utilities": "__fou__",
          "@fiftyone/spaces": "__fosp__",
          "@fiftyone/aggregations": "__foa__",
          "@fiftyone/core": "__focore__",
          "@mui/material": "__mui__",
        },
      },
    },
  },
  define: {
    "process.env.NODE_ENV": '"development"',
  },
  optimizeDeps: {
    exclude: ["react", "react-dom"],
  },
});
