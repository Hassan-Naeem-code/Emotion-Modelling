import { defineConfig } from "vitest/config";

// The abstention tests are pure logic (no DOM/camera/model), so the default
// node environment is enough and keeps the suite fast.
export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
