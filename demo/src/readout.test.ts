import { describe, expect, it } from "vitest";
import { coarseLabel } from "./readout";

describe("coarseLabel (derived VA heuristic)", () => {
  it("returns neutral near the origin", () => {
    expect(coarseLabel(0.05, -0.05)).toBe("neutral");
  });
  it("maps the four circumplex quadrants", () => {
    expect(coarseLabel(0.6, 0.6)).toBe("excited / happy");
    expect(coarseLabel(-0.6, 0.6)).toBe("tense / angry");
    expect(coarseLabel(-0.6, -0.6)).toBe("sad / bored");
    expect(coarseLabel(0.6, -0.6)).toBe("calm / content");
  });
});
