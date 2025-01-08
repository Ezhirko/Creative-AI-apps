import gradio as gr
from BEPTokenizer import BEPTokenizer

# Initialize the BEPTokenizer
file_path = "data/tamil_article_corpus.txt"

# Read the corpus file
try:
    with open(file_path, "r", encoding="utf-8") as file:
        corpus_content = file.read()
except FileNotFoundError:
    corpus_content = ""
    print(f"Error: The file at {file_path} was not found.")
except Exception as e:
    corpus_content = ""
    print(f"An error occurred while reading the file: {e}")

max_vocab_size = 5000
tamil_tokenizer = BEPTokenizer(corpus_content, max_vocab_size)
tamil_tokenizer.train_BPE_Tokenizer()

# Define the Gradio app functions
def encode_text(text):
    encoded_tokens = tamil_tokenizer.encode(text)
    encoded_tokens_display = ", ".join(map(str, encoded_tokens))
    return encoded_tokens_display

def decode_tokens(tokens):
    try:
        token_list = list(map(int, tokens.split(",")))  # Convert input string to list of integers
        decoded_text = tamil_tokenizer.decode(token_list)
        return decoded_text
    except Exception as e:
        return f"Error decoding tokens: {e}"

# Gradio app interface
with gr.Blocks() as app:
    gr.Markdown(
    """
    <h1 style="text-align: center; font-size: 2.5em;">தமிழ் உரை டோக்கனைசர் (பிபிஇ என்கோடிங் மற்றும் டிகோடிங்)</h1>
    <p>Tamil Text Tokenizer (BPE Encoding and Decoding)</p>
    <h2>Steps to Use:</h2>
    <ol>
        <li>Enter text in Tamil in the provided text box.</li>
        <li>Click the <strong>Encode</strong> button to convert the text into encoded tokens.</li>
        <li>Copy the encoded tokens displayed below the text box.</li>
        <li>Paste the copied tokens into the text box for decoding.</li>
        <li>Click the <strong>Decode</strong> button to verify if the decoded text matches your original input.</li>
    </ol>
    """,
    elem_id="title"
)

    
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Encode Text")
            input_text = gr.Textbox(
                label="Enter Tamil Text", 
                lines=5, 
                placeholder="உங்கள் மெர்சிடிஸ் பென்ஸ் கார் அழகாக இருக்கிறத."
            )
            encode_button = gr.Button("Encode")
            encoded_output = gr.Textbox(
                label="Encoded Tokens:", 
                lines=5, 
                interactive=False, 
                placeholder="Encoded tokens will appear here."
            )
        
        with gr.Column():
            gr.Markdown("#### Decode Tokens")
            token_input = gr.Textbox(
                label="Enter Tokens (comma-separated)", 
                lines=5, 
                placeholder="Example: 107,123,256"
            )
            decode_button = gr.Button("Decode")
            decoded_output = gr.Textbox(
                label="Decoded Text:", 
                lines=5, 
                interactive=False, 
                placeholder="Decoded text will appear here."
            )
    
    # Connect functions to buttons
    encode_button.click(
        encode_text, 
        inputs=input_text, 
        outputs=encoded_output
    )
    decode_button.click(
        decode_tokens, 
        inputs=token_input, 
        outputs=decoded_output
    )

# Run the app
app.launch()