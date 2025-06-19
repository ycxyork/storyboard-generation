# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import storyboard_generation

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
        text: str = Input(description="Prompt for Storyboard generation"),
        mask_scale: float = Input(description="Factor to scale mask", ge=0, le=2, default=1),
        aspect_ratio: str = Input(description="Ratio for generation image",default="512 x 1024"),
        seed: int = Input(description="Seed of generation", default=42),
        
    ) -> str:
        """Run a single prediction on the model"""

        storyboard_generationinit(seed)
        width, height = storyboard_generation.parse_aspect_ratio(aspect_ratio)
        prompt = storyboard_generation.process_prompt(text)
        image_path = storyboard_generation.base_generation(prompt[0], width, height)
        _, mask_path = storyboard_generation.detect_and_generate_mask(image_path, mask_scale=mask_scale)
        image_list = storyboard_generation.inpaint(prompt[1:4], image_path, mask_path)
        return ",".join(image_list)
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)

