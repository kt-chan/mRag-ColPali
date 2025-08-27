import base64
import io
import json
import os
import cv2
import torch
from dotenv import load_dotenv
from pathlib import Path
import gradio as gr
from pdf2image import convert_from_path
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2, ColQwen2Processor
from models.chatglm import ChatGLM
import threading

COLORS = ["#4285f4", "#db4437", "#f4b400", "#0f9d58", "#e48ef1"]
mock_image = Image.new("RGB", (448, 448), (255, 255, 255))

# CSS for emojis
css = """
.emoji {
    width: 20px !important;
    height: 20px !important;
    display: inline-block !important;
    vertical-align: middle !important;
    margin: 0 4px !important;
}
"""


class ColPaliApp:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ColPaliApp, cls).__new__(cls)
                    cls.processor = None
                    cls.model = None
                    cls.device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    cls.models_loaded = False
                    cls.model_name = "vidore/colqwen2-v1.0"
                    cls.hf_home = os.getenv(
                        "HF_HOME", "C:/Program Files/poppler/Library/bin"
                    )
                    cls.popper_path = os.getenv("POPPER_PATH", "D:/models/huggingface")
                    cls.vlm_api_key = os.getenv("VLM_API_KEY", None)
                    cls.reload()
        return cls._instance

    @classmethod
    def instance(cls):
        return cls()

    @classmethod
    def reload(cls):

        if cls.processor is not None and cls.model is not None:
            return

        model_path = cls.model_name
        print(f"torch.version: {torch.__version__}")
        print(f"torch.version.cuda: {torch.version.cuda}")
        print(f"cuda availability: {torch.cuda.is_available()}")
        print(f"cuda device Name: {torch.cuda.get_device_name()}")
        print(f"Loading ColPali model on device: {cls.device}")
        print(f"Loading ColPali model on path: {model_path}")

        try:
            cls.model = ColQwen2.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=cls.device,
                attn_implementation=(
                    "flash_attention_2" if is_flash_attn_2_available() else None
                ),
            ).eval()

            if cls.model is None:
                raise ValueError(f"model: {model_path} failed to load")

            cls.processor = ColQwen2Processor.from_pretrained(model_path)
            if cls.processor is None:
                raise ValueError(f"Processor: {model_path} failed to load")

            cls.models_loaded = True
            print(f"ColPali model loaded successfully with Device: {cls.device}!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")

    @classmethod
    def encode_image(cls, image: Image):
        with io.BytesIO() as buffer:
            image.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def index(self, file, ds):
        images = []
        for f in file:
            images.extend(convert_from_path(f, poppler_path=self.popper_path))

        dataloader = DataLoader(
            images,
            batch_size=4,
            shuffle=False,
            collate_fn=lambda x: self.processor.process_images(x),
        )

        for batch_doc in tqdm(dataloader):
            with torch.no_grad():
                batch_doc = {k: v.to(self.device) for k, v in batch_doc.items()}
                embeddings_doc = self.model(**batch_doc)
            ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
        return f"Uploaded and converted {len(images)} pages", ds, images

    def search(self, query: str, ds, images):
        qs = []
        with torch.no_grad():
            batch_query = self.processor.process_queries([query])
            batch_query = {k: v.to(self.device) for k, v in batch_query.items()}
            embeddings_query = self.model(**batch_query)
            qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

        scores = self.processor.score_multi_vector(qs, ds)
        best_page = int(scores.argmax(axis=1).item())
        return f"The most relevant page is {best_page}", images[best_page]

    def get_answer(self, prompt: str, image: Image):

        # Initialize client
        vlmClient = ChatGLM(self.vlm_api_key)

        # Example request
        # Analysis of main volatile components in the Yinqiaosan decoction samples with different time by gas chromatography‑mass spectrometer
        try:
            base64_image = self.encode_image(image)
            print("Image decoded, and waiting API Response:")
            response = vlmClient.chat_completion(
                image_url=f"data:image/jpeg;base64,{base64_image}",
                text_prompt=f"{prompt}",
            )
            print("Waiting API Response:")
            return response
        except Exception as e:
            print(f"VLM Calling Error: {str(e)}")

    def search_with_llm(self, query, ds, images):
        search_message, best_image = self.search(query, ds, images)
        answer = self.get_answer(
            "What is shown in this image, analyze and provide some interpretation?",
            best_image,
        )
        return search_message, best_image, answer


# Gradio Interface
with gr.Blocks(css=css) as demo:
    gr.Markdown("# ColPali: Efficient Document Retrieval")
    gr.Markdown(
        '## <img draggable="false" role="img" class="emoji" alt="1️⃣" src="https://s.w.org/images/core/emoji/16.0.1/svg/31-20e3.svg  "> Upload PDFs'
    )
    file = gr.File(file_types=[".pdf"], file_count="multiple", type="filepath")
    gr.Markdown(
        '## <img draggable="false" role="img" class="emoji" alt="2️⃣" src="https://s.w.org/images/core/emoji/16.0.1/svg/32-20e3.svg  "> Index the PDFs'
    )
    convert_button = gr.Button("Convert and upload")
    message = gr.Textbox("Files not yet uploaded")
    embeds = gr.State(value=[])
    imgs = gr.State(value=[])

    convert_button.click(
        lambda file, ds: ColPaliApp.instance().index(file, ds),
        inputs=[file, embeds],
        outputs=[message, embeds, imgs],
    )

    gr.Markdown(
        '## <img draggable="false" role="img" class="emoji" alt="3️⃣" src="https://s.w.org/images/core/emoji/16.0.1/svg/33-20e3.svg  "> Search'
    )
    query = gr.Textbox(placeholder="Enter your query to match", lines=10)
    search_button = gr.Button("Search")

    gr.Markdown(
        '## <img draggable="false" role="img" class="emoji" alt="4️⃣" src="https://s.w.org/images/core/emoji/16.0.1/svg/34-20e3.svg  "> ColPali Retrieval'
    )
    message2 = gr.Textbox("Most relevant image is...")
    output_img = gr.Image()
    output_text = gr.Textbox(label="LLM Response")

    search_button.click(
        lambda query, ds, images: ColPaliApp.instance().search_with_llm(
            query, ds, images
        ),
        inputs=[query, embeds, imgs],
        outputs=[message2, output_img, output_text],
    )

if __name__ == "__main__":
    load_dotenv()
    ColPaliApp.instance()
    demo.queue(max_size=10).launch(debug=True, share=True)
