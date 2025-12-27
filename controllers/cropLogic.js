import { PythonShell } from "python-shell";
import path from "path";

export async function recommendCrop(req, res) {
  try {
    const options = {
      scriptPath: path.join(process.cwd(), "python"),
      args: [JSON.stringify(req.body)]
    };

    PythonShell.run("crop_predict.py", options, (err, result) => {
      if (err) {
        console.error("ML ERROR:", err);
        return res.status(500).json({ error: "ML prediction failed" });
      }

      const parsed = JSON.parse(result[0]);

      res.json({
        crop: parsed.crop,
        confidence: parsed.confidence,
        expectedYield: parsed.expectedYield,
        reasoning: parsed.reasoning
      });
    });
  } catch (err) {
    console.error("CROP CONTROLLER ERROR:", err.message);
    res.status(500).json({ error: "Crop prediction failed" });
  }
}
