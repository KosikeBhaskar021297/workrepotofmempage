import speech_recognition as sr
import openai
from fpdf import FPDF

# Set up your OpenAI API credentials
openai.api_key = "YOUR_OPENAI_API_KEY"

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Word Details", 0, 1, "C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

# Function to transcribe speech
def transcribe_speech():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Transcribing...")
        text = recognizer.recognize_google_cloud(audio)
        return text
    except sr.UnknownValueError:
        print("Unable to transcribe speech.")
    except sr.RequestError as e:
        print(f"Error: {e}")

# Call the function to transcribe speech
transcribed_text = transcribe_speech()
print("Transcribed Text:", transcribed_text)

# Query ChatGPT for word details
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=f"What can you tell me about {transcribed_text}?",
    max_tokens=50
)
word_details = response.choices[0].text.strip()

# Print the word details from ChatGPT
print("Word Details:", word_details)

# Save word details in a PDF
pdf = PDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(0, 10, word_details, ln=True)
pdf.output("word_details.pdf")

import pywhatkit as pwk

# Recipient's phone number (with country code)
recipient_phone_number = '+918142873722'

# Message to send
message = 'Generated text from code:\n' + word_details
     
# Send the message using pywhatkit
pwk.sendwhatmsg_instantly(recipient_phone_number, message)

print("WhatsApp message sent using pywhatkit.")
