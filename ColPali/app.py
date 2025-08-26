import os, torch
import gradio as gr
from pdf2image import convert_from_path
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2, ColQwen2Processor

# Global variables to store the loaded model and processor
processor = None
model = None
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# model_dir = Path("D:\\models\\modelscope\\hub\\models")\
model_dir = ""
model_name = "vidore/colqwen2-v1.0"
models_loaded = False  # Flag to track if models are loaded


def load_models():

    global processor, model, models_loaded

    if models_loaded:  # Skip if already loaded
        return

    model_path = model_name
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())  # Should return True
    print(torch.cuda.get_device_name(0))  # Should display your GPU name.
    print(f"Loading ColPali model on device: {device}")
    print(f"Loading ColPali model on path: {model_path}")

    try:
        # Load model using ModelScope
        model = ColQwen2.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation=(
                "flash_attention_2" if is_flash_attn_2_available() else None
            ),
        ).eval()

        if model is None:
            raise ValueError(f"model: {model_path} failed to load")
        
        # For processor, you might need to check if ModelScope has a compatible version
        # If not, you can still use the original processor but ensure model files are downloaded
        processor = ColQwen2Processor.from_pretrained(model_path)
        
        if processor is None:
            raise ValueError(f"Processor: {model_path} failed to load")
    
        models_loaded = True  # Set flag to True after loading
        print(f"ColPali model loaded successfully with Device: {device}!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")


def index(file, ds):
    global processor, model
    load_models()  # Ensure models are loaded

    images = []
    for f in file:
        images.extend(convert_from_path(f, poppler_path=r'C:\\Program Files\\poppler\\Library\\bin'))

    # run inference - docs
    dataloader = DataLoader(
        images,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x),
    )
    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
    return f"Uploaded and converted {len(images)} pages", ds, images


def search(query: str, ds, images):
    global processor, model
    load_models()  # Ensure models are loaded

    qs = []
    with torch.no_grad():
        batch_query = processor.process_queries([query], mock_image)
        batch_query = {k: v.to(device) for k, v in batch_query.items()}
        embeddings_query = model(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

    # run evaluation
    scores = processor.score_multi_vector(qs, ds)
    best_page = int(scores.argmax(axis=1).item())
    return f"The most relevant page is {best_page}", images[best_page]


COLORS = ["#4285f4", "#db4437", "#f4b400", "#0f9d58", "#e48ef1"]

mock_image = Image.new("RGB", (448, 448), (255, 255, 255))

# Add this CSS style to make emojis small
css = """
.emoji {
    width: 20px !important;
    height: 20px !important;
    display: inline-block !important;
    vertical-align: middle !important;
    margin: 0 4px !important;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# ColPali: Efficient Document Retrieval with Vision Language Models ")

    gr.Markdown(
        '## <img draggable="false" role="img" class="emoji" alt="1️⃣" src="https://s.w.org/images/core/emoji/16.0.1/svg/31-20e3.svg "> Upload PDFs'
    )
    file = gr.File(
        file_types=[".pdf"],
        file_count="multiple",
        type="filepath",  # or "binary" if you need bytes
    )
    gr.Markdown(
        '## <img draggable="false" role="img" class="emoji" alt="2️⃣" src="https://s.w.org/images/core/emoji/16.0.1/svg/32-20e3.svg "> Index the PDFs and upload'
    )
    gr.Markdown(
        '<img draggable="false" role="img" class="emoji" alt="🔄" src="https://s.w.org/images/core/emoji/16.0.1/svg/1f504.svg "> Convert and upload'
    )
    convert_button = gr.Button("Convert and upload")
    message = gr.Textbox("Files not yet uploaded")
    embeds = gr.State(value=[])
    imgs = gr.State(value=[])

    # Define the actions for conversion
    convert_button.click(index, inputs=[file, embeds], outputs=[message, embeds, imgs])

    gr.Markdown(
        '## <img draggable="false" role="img" class="emoji" alt="3️⃣" src="https://s.w.org/images/core/emoji/16.0.1/svg/33-20e3.svg "> Search'
    )
    query = gr.Textbox(placeholder="Enter your query to match", lines=10)

    gr.Markdown(
        '<img draggable="false" role="img" class="emoji" alt="🔍" src="https://s.w.org/images/core/emoji/16.0.1/svg/1f50d.svg "> Search'
    )
    search_button = gr.Button("Search")

    gr.Markdown(
        '## <img draggable="false" role="img" class="emoji" alt="4️⃣" src="https://s.w.org/images/core/emoji/16.0.1/svg/34-20e3.svg "> ColPali Retrieval'
    )

    message2 = gr.Textbox("Most relevant image is...")
    output_img = gr.Image()
    output_text = gr.Textbox(label="LLM Analysis")  # Add output for LLM response

    # Add output text component for LLM response
    output_text = gr.Textbox(label="LLM Response")

    def get_answer(prompt: str, image: Image):
        ### Update to use vlm
        # response = gemini_flash.generate_content([prompt, image])
        # return response.text
        return None

    # Function to combine retrieval and LLM call
    def search_with_llm(
        query,
        ds,
        images,
        prompt="What is shown in this image, analyse and provide some interpretation? Format the answer in a neat 500 words summary.",
    ):
        # Step 1: Search the best image based on query
        search_message, best_image = search(query, ds, images)

        # Step 2: Generate an answer using LLM
        answer = get_answer(prompt, best_image)

        return search_message, best_image, answer

    # Action for search button
    search_button.click(
        search_with_llm,
        inputs=[query, embeds, imgs],
        outputs=[message2, output_img, output_text],
    )

if __name__ == "__main__":
    load_models()  # Pre-load models on startup
    demo.queue(max_size=10).launch(debug=True, share=True)
