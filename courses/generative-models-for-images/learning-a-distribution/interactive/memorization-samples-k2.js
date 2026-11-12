/* Samples of the circle generators of Chapter 1 with two latent dimensions, widths 16 and 1024 side by side, for 12 or
   200 training points (one draw, seed 0). Drawing in circle-samples.js, without the curve of the generator,
   which exists only for one latent dimension.
   Data: memorization-samples-k2.json, written by _dev/figures/memorization_sweeps.py post.
   Static version: figures/memorization-samples-k2.png. */
import { figure } from "./circle-samples.js";

export default figure({
	file: "memorization-samples-k2.json",
	latent: 2,
	notes: {
		12: "The wide generator puts most of its samples on the training points and on the chords between them.",
		200: "With 200 points, both generators follow the circle.",
	},
});
