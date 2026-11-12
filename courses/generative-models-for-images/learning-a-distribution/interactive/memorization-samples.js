/* Samples of the circle generators of Chapter 1 with one latent dimension, widths 16 and 1024 side by side, for 12 or
   200 training points (one draw, seed 1), with the curve of each generator. Drawing in circle-samples.js.
   Data: memorization-samples.json, written by _dev/figures/memorization_sweeps.py post.
   Static version: figures/memorization-samples.png. */
import { figure } from "./circle-samples.js";

export default figure({
	file: "memorization-samples.json",
	latent: 1,
	notes: {
		12: "The wide generator runs along the chords from one training point to the next.",
		200: "With 200 points, both generators follow the circle.",
	},
});
