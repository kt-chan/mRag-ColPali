import io
import os
import torch
import base64
import gradio as gr
import threading
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2, ColQwen2Processor
from models.chatglm import ChatGLM
from utility.video_processor import VideoProcessor

# Global Constants
COLORS: List[str] = ["#4285f4", "#db4437", "#f4b400", "#0f9d58", "#e48ef1"]
MOCK_IMAGE: Image.Image = Image.new("RGB", (448, 448), (255, 255, 255))
DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_NAME: str = "vidore/colqwen2-v1.0"

# CSS for emojis
CSS_CONFIG: str = """
.emoji {
    width: 20px !important;
    height: 20px !important;
    display: inline-block !important;
    vertical-align: middle !important;
    margin: 0 4px !important;
}
"""


if gr.NO_RELOAD:
    print(f"Loading ColPali model on device: {DEVICE}")
    print(f"Loading ColPali model on path: {MODEL_NAME}")
    try:
        MODEL = ColQwen2.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map=DEVICE,
            attn_implementation=(
                "flash_attention_2" if is_flash_attn_2_available() else None
            ),
        ).eval()

        PROCESSOR = ColQwen2Processor.from_pretrained(MODEL_NAME)
        MODEL_LOADED = True
        print(f"ColPali model loaded successfully with Device: {DEVICE}!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        MODEL = None
        PROCESSOR = None
        MODEL_LOADED = False


class ColPaliApp:
    _instance: Optional["ColPaliApp"] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls) -> "ColPaliApp":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ColPaliApp, cls).__new__(cls)
                    cls.model: Optional[ColQwen2] = MODEL
                    cls.processor: Optional[ColQwen2Processor] = PROCESSOR
                    cls.device: str = DEVICE
                    cls.models_loaded: bool = MODEL_LOADED
                    cls.model_name: str = "vidore/colqwen2-v1.0"
                    cls.hf_home: str = os.getenv(
                        "HF_HOME", "C:/Program Files/poppler/Library/bin"
                    )
                    cls.popper_path: str = os.getenv(
                        "POPPER_PATH", "D:/models/huggingface"
                    )
                    cls.vlm_api_key: Optional[str] = os.getenv("VLM_API_KEY", None)
                    cls.video_processor: VideoProcessor = VideoProcessor()
        return cls._instance

    def __init__(self) -> None:
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._initialized = True

    @classmethod
    def reload(cls) -> None:
        global MODEL_LOADED, MODEL, PROCESSOR
        if not MODEL_LOADED or MODEL is None or PROCESSOR is None:
            print(f"Loading ColPali model on device: {DEVICE}")
            print(f"Loading ColPali model on path: {MODEL_NAME}")

            try:
                MODEL = ColQwen2.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.bfloat16,
                    device_map=DEVICE,
                    attn_implementation=(
                        "flash_attention_2" if is_flash_attn_2_available() else None
                    ),
                ).eval()

                PROCESSOR = ColQwen2Processor.from_pretrained(MODEL_NAME)
                MODEL_LOADED = True
                cls.models_loaded = MODEL_LOADED
                cls.model = MODEL
                cls.processor = PROCESSOR
                cls.device = DEVICE
                print(f"ColPali model loaded successfully with Device: {DEVICE}!")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                MODEL = None
                PROCESSOR = None
                MODEL_LOADED = False

    @classmethod
    def sample_video(cls, filename: str) -> List[Image.Image]:
        try:
            return cls.video_processor.sample_frames(
                video_path=filename, interval_seconds=5
            )
        except Exception as e:
            print(f"Error sampling video {filename}: {str(e)}")
            return []

    @classmethod
    def encode_image(cls, image: Image.Image) -> str:
        try:
            with io.BytesIO() as buffer:
                image.save(buffer, format="JPEG")
                return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"Error encoding image: {str(e)}")
            return ""

    def check_file_extension(self, file_path: str) -> str:
        try:
            return Path(file_path).suffix[1:].upper()
        except Exception as e:
            print(f"Error checking file extension {file_path}: {str(e)}")
            return ""

    def index(
        self, files: List[str], ds: List[Any]
    ) -> Tuple[str, List[Any], List[Image.Image]]:
        try:
            pdf_files: List[str] = []
            video_files: List[str] = []

            for f in files:
                file_ext = self.check_file_extension(f)
                if file_ext == "PDF":
                    pdf_files.append(f)
                elif file_ext == "MP4":
                    video_files.append(f)

            if pdf_files:
                return self.index_images_from_pdf(pdf_files, ds)
            elif video_files:
                return self.index_videos(video_files, ds)
            else:
                return "No valid files found", ds, []
        except Exception as e:
            print(f"Error in indexing: {str(e)}")
            return f"Error during indexing: {str(e)}", ds, []

    def index_videos(
        self, files: List[str], ds: List[Any]
    ) -> Tuple[str, List[Any], List[Image.Image]]:
        if not self.models_loaded or self.model is None or self.processor is None:
            print(f"Model not loaded properly, reloading ...")
            self.reload()
        video_files = []
        for file in files:
            video_files.extend(self.sample_video(str(Path(file).resolve())))
        videos = []
        for video_file in video_files:
            with Image.open(video_file) as img:
                # Load image data immediately to decouple from file handle
                img.load()
                videos.append(img)
        return self.index_images(videos, ds)

    def index_images_from_pdf(
        self, files: List[str], ds: List[Any]
    ) -> Tuple[str, List[Any], List[Image.Image]]:
        if not self.models_loaded or self.model is None or self.processor is None:
            print("Model not loaded properly, reloading ...")
            self.reload()

        images: List[Image.Image] = []
        for f in files:
            try:
                images.extend(convert_from_path(f, poppler_path=self.popper_path))
            except Exception as e:
                print(f"Error converting PDF {f}: {str(e)}")
                continue

        return self.index_images(images, ds)

    def index_images(
        self, images: List[Image.Image], ds: List[Any]
    ) -> Tuple[str, List[Any], List[Image.Image]]:
        if not images:
            return "No images to process", ds, []

        try:
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
        except Exception as e:
            print(f"Error during image indexing: {str(e)}")
            return f"Error during image processing: {str(e)}", ds, images

    def search(
        self, query: str, ds: List[Any], images: List[Image.Image]
    ) -> Tuple[str, Image.Image]:
        if not self.models_loaded or self.model is None or self.processor is None:
            print("Model not loaded properly, reloading ...")
            self.reload()

        try:
            qs: List[Any] = []
            with torch.no_grad():
                batch_query = self.processor.process_queries([query])
                batch_query = {k: v.to(self.device) for k, v in batch_query.items()}
                embeddings_query = self.model(**batch_query)
                qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

            scores = self.processor.score_multi_vector(qs, ds)
            best_page = int(scores.argmax(axis=1).item())
            return f"The most relevant page is {str(best_page)}", images[best_page]
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return f"Search error: {str(e)}", MOCK_IMAGE

    def get_answer(self, prompt: str, image: Image.Image) -> str:
        if not self.vlm_api_key:
            return "VLM API key not configured"

        try:
            vlmClient = ChatGLM(self.vlm_api_key)
            base64_image = self.encode_image(image)
            print("Image decoded, and waiting API Response:")
            response = vlmClient.chat_completion(
                image_url=f"data:image/jpeg;base64,{base64_image}",
                text_prompt=f"{prompt}",
            )
            print("Received API Response:")
            return response
        except Exception as e:
            print(f"VLM Calling Error: {str(e)}")
            return f"Error: {str(e)}"

    def search_with_llm(
        self, query: str, ds: List[Any], images: List[Image.Image]
    ) -> Tuple[str, Image.Image, str]:
        search_message, best_image = self.search(query, ds, images)
        answer = self.get_answer(
            f"What is shown in this image, analyze and provide some interpretation? And provide a concise answer to this question: {query}",
            best_image,
        )
        return search_message, best_image, answer


app = ColPaliApp()

# Gradio Interface
with gr.Blocks(css=CSS_CONFIG) as demo:
    gr.Markdown("# AnyVision: Efficient Visual Retrieval for Anything")

    # Upload Section
    gr.Markdown(
        '## <img draggable="false" role="img" class="emoji" alt="1️⃣" src="https://s.w.org/images/core/emoji/16.0.1/svg/31-20e3.svg"> Upload PDFs'
    )
    file = gr.File(
        file_types=[".pdf", ".mp4"],
        file_count="multiple",
        type="filepath",
        elem_id="pdf_upload",
        label="Upload PDFs",
    )

    # Index Section
    gr.Markdown(
        '## <img draggable="false" role="img" class="emoji" alt="2️⃣" src="https://s.w.org/images/core/emoji/16.0.1/svg/32-20e3.svg"> Index the PDFs'
    )
    convert_button = gr.Button("Convert and upload", elem_id="convert_btn")
    message = gr.Textbox(
        "Files not yet uploaded",
        elem_id="status_msg",
        label="Status",
    )
    embeds = gr.State(value=[])
    imgs = gr.State(value=[])

    # Search Section
    gr.Markdown(
        '## <img draggable="false" role="img" class="emoji" alt="3️⃣" src="https://s.w.org/images/core/emoji/16.0.1/svg/33-20e3.svg"> Your Question?'
    )
    query = gr.Textbox(
        placeholder="Enter your query to match",
        lines=10,
        elem_id="query_input",
        label="Query",
    )
    search_button = gr.Button("Search", elem_id="search_btn")

    # Results Section
    gr.Markdown(
        '## <img draggable="false" role="img" class="emoji" alt="4️⃣" src="https://s.w.org/images/core/emoji/16.0.1/svg/34-20e3.svg"> AnyVision Retrieval ...'
    )
    message2 = gr.Textbox(
        "Most relevant image is...",
        elem_id="retrieval_msg",
        label="Retrieval Result",
    )
    output_img = gr.Image(elem_id="output_image", label="Retrieved Image")
    output_text = gr.Textbox(label="LLM Response", elem_id="llm_response")

    # Event handlers
    convert_button.click(
        app.index,
        inputs=[file, embeds],
        outputs=[message, embeds, imgs],
    )

    search_button.click(
        app.search_with_llm,
        inputs=[query, embeds, imgs],
        outputs=[message2, output_img, output_text],
    )


if __name__ == "__main__":
    load_dotenv()
    demo.queue(max_size=10).launch(debug=True, share=False, pwa=True)
