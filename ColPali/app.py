import io, os, torch, base64, json, cv2
import gradio as gr
import threading
from dotenv import load_dotenv
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
from IPython.display import Video
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2, ColQwen2Processor
from models.chatglm import ChatGLM
from utility.video_processor import VideoProcessor

# Global Variable
COLORS = ["#4285f4", "#db4437", "#f4b400", "#0f9d58", "#e48ef1"]
MOCK_IMAGE = Image.new("RGB", (448, 448), (255, 255, 255))
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "vidore/colqwen2-v1.0"

# CSS for emojis
CSS_CONFIG = """
.emoji {
    width: 20px !important;
    height: 20px !important;
    display: inline-block !important;
    vertical-align: middle !important;
    margin: 0 4px !important;
}
"""


# Initialize model and processor once using gr.NO_RELOAD
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
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ColPaliApp, cls).__new__(cls)
                    cls.model = MODEL
                    cls.processor = PROCESSOR
                    cls.device = DEVICE
                    cls.models_loaded = MODEL_LOADED
                    cls.model_name = "vidore/colqwen2-v1.0"
                    cls.hf_home = os.getenv(
                        "HF_HOME", "C:/Program Files/poppler/Library/bin"
                    )
                    cls.popper_path = os.getenv("POPPER_PATH", "D:/models/huggingface")
                    cls.vlm_api_key = os.getenv("VLM_API_KEY", None)
        return cls._instance

    @classmethod
    def instance(cls):
        return cls()

    @classmethod
    def reload(cls):
        # No-op since models are loaded in gr.NO_RELOAD block
        global MODEL_LOADED, MODEL, PROCESSOR, DEVICE
        if MODEL_LOADED == False or MODEL is None or PROCESSOR is None:
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
    def sample_video(cls, filename):
        video_client = VideoProcessor()
        video_frame_outputs = []
        video_frame_outputs.extend(
            video_client.sample_frames(video_path=filename, interval_seconds=5)
        )
        return video_frame_outputs

    @classmethod
    def encode_image(cls, image: Image):
        with io.BytesIO() as buffer:
            image.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def check_file_extension(sefl, file_path):
        # Get the file extension
        _, file_extension = os.path.splitext(file_path)
        # Normalize the extension to uppercase for comparison
        file_extension = file_extension.upper()
        return str(file_extension[1:])

    def index(self, files, ds):
        pdf_files = []
        video_files = []

        for f in files:
            file_ext = self.check_file_extension(f)
            if file_ext.upper() == "PDF":
                pdf_files.append(f)
            if file_ext.upper() == "MP4":
                video_files.append(f)

        if len(pdf_files) > 0:
            return self.index_images_from_pdf(pdf_files, ds)

        if len(video_files) > 0:
            return self.index_videos(video_files, ds)

    def index_videos(self, files, ds):
        if not self.models_loaded or self.model is None or self.processor is None:
            print(f"Model not loaded properly, reloading ...")
            self.reload()
        videos = []
        for file in files:
            videos.extend(self.sample_video(str(file.name)))

        output = self.index_images(videos, ds)
        return output

    def index_images_from_pdf(self, files, ds):
        if not self.models_loaded or self.model is None or self.processor is None:
            print(f"Model not loaded properly, reloading ...")
            self.reload()

        images = []
        for f in files:
            images.extend(convert_from_path(f, poppler_path=self.popper_path))

        return self.index_images(images, ds)

    def index_images(self, images, ds):

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
        if not self.models_loaded or self.model is None or self.processor is None:
            print(f"Model not loaded properly, reloading ...")
            self.reload()

        qs = []
        with torch.no_grad():
            batch_query = self.processor.process_queries([query])
            batch_query = {k: v.to(self.device) for k, v in batch_query.items()}
            embeddings_query = self.model(**batch_query)
            qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

        scores = self.processor.score_multi_vector(qs, ds)
        best_page = int(scores.argmax(axis=1).item())
        return f"The most relevant page is {str(best_page)}", images[best_page]

    def get_answer(self, prompt: str, image: Image):
        if not self.vlm_api_key:
            return "VLM API key not configured"

        vlmClient = ChatGLM(self.vlm_api_key)
        try:
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

    def search_with_llm(self, query, ds, images):
        if not self.models_loaded or self.model is None or self.processor is None:
            print(f"Model not loaded properly, reloading ...")
            self.reload()

        search_message, best_image = self.search(query, ds, images)
        answer = self.get_answer(
            f"What is shown in this image, analyze and provide some interpretation? And provide a concise answer to this question: {query}",
            best_image,
        )
        return search_message, best_image, answer


# Gradio Interface
with gr.Blocks(css=CSS_CONFIG) as demo:
    gr.Markdown("# AnyVision: Efficient Visual Retrieval for Anything")

    # Upload Section
    gr.Markdown(
        '## <img draggable="false" role="img" class="emoji" alt="1️⃣" src="https://s.w.org/images/core/emoji/16.0.1/svg/31-20e3.svg  "> Upload PDFs'
    )
    file = gr.File(
        file_types=[".pdf", ".mp4"],
        file_count="multiple",
        type="filepath",
        elem_id="pdf_upload",  # Unique ID
        label="Upload PDFs",  # Explicit label
    )

    # Index Section
    gr.Markdown(
        '## <img draggable="false" role="img" class="emoji" alt="2️⃣" src="https://s.w.org/images/core/emoji/16.0.1/svg/32-20e3.svg  "> Index the PDFs'
    )
    convert_button = gr.Button("Convert and upload", elem_id="convert_btn")  # Unique ID
    message = gr.Textbox(
        "Files not yet uploaded",
        elem_id="status_msg",  # Unique ID
        label="Status",  # Explicit label
    )
    embeds = gr.State(value=[])
    imgs = gr.State(value=[])

    # Search Section
    gr.Markdown(
        '## <img draggable="false" role="img" class="emoji" alt="3️⃣" src="https://s.w.org/images/core/emoji/16.0.1/svg/33-20e3.svg  "> Search'
    )
    query = gr.Textbox(
        placeholder="Enter your query to match",
        lines=10,
        elem_id="query_input",  # Unique ID
        label="Query",  # Explicit label
    )
    search_button = gr.Button("Search", elem_id="search_btn")  # Unique ID

    # Results Section
    gr.Markdown(
        '## <img draggable="false" role="img" class="emoji" alt="4️⃣" src="https://s.w.org/images/core/emoji/16.0.1/svg/34-20e3.svg  "> ColPali Retrieval'
    )
    message2 = gr.Textbox(
        "Most relevant image is...",
        elem_id="retrieval_msg",  # Unique ID
        label="Retrieval Result",  # Explicit label
    )
    output_img = gr.Image(
        elem_id="output_image", label="Retrieved Image"  # Unique ID  # Explicit label
    )
    output_text = gr.Textbox(
        label="LLM Response", elem_id="llm_response"  # Explicit label  # Unique ID
    )

    # Event handlers remain unchanged
    convert_button.click(
        lambda file, ds: ColPaliApp.instance().index(file, ds),
        inputs=[file, embeds],
        outputs=[message, embeds, imgs],
    )

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
    demo.queue(max_size=10).launch(debug=True, share=False, pwa=True)
