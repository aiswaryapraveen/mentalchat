import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
import random
import contractions

# Load the sentiment analysis model
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Response categories
responses = {
    "greeting": ["Hey there! 😊 How are you feeling today?", "Hi! Hope you're doing well. 💙", "Hello! What’s on your mind?"],
    "motivation": ["You’ve got this! Stay strong and keep going! 💪🚀", "Stay focused, break your tasks into small steps, and take it one step at a time! 📚", "Every step you take brings you closer to success. Keep pushing! 🔥"],
    "venting": ["You deserve love no matter what head space you're in. If you don't have people to remind you of that when you need it, just know i am always here for each and every one of you.","I'm here for you. You can share anything with me. 💙", "It's okay to feel this way. You're not alone. I'm here to listen. 💙", "Take a deep breath. I'm here for you. Let's talk."],
    "stress": ["Life is overwhelming. You might be in school, you might have a job, or even just the demanding expectations in everyday life can cause crazy amounts of stress, everyone needs a break every now and then. You're doing great! im so proud of you","Deep breaths. You’ve handled tough situations before, and you’ll get through this too. 💙", "Try taking a short break or going for a walk. You’re doing great. 🌿", "Remember to take things one step at a time. You’re stronger than you think!","Life is crazy sometimes, you're not being dramatic,you're doing just fine! Take a few minutes if you can & just breathe, take some time to ground yourself and be aware of your surroundings."],
    "gratitude": ["That’s so kind of you! I appreciate you too! 😊", "You're awesome! Thank you for sharing positivity. 💙", "Your kindness makes the world brighter. Thank you! 😊"],
    "casual": [
        "I’m doing great! Thanks for asking. How about you? 😊",
        "I'm here and ready to chat! What's on your mind? 💙",
        "Feeling good! How’s your day going? 🌞",
        ],
    "joke": ["Why don’t skeletons fight each other? They don’t have the guts! 😆", "What do you call fake spaghetti? An impasta! 🍝😂", "Why did the scarecrow win an award? Because he was outstanding in his field! 🌾🤣"],
    "self_care": ["Taking care of yourself is important! Try deep breathing, listening to music, or taking a short walk. 🌿", "Self-care tip: Stay hydrated, take breaks, and do something you love today! 💙", "A little self-care goes a long way! Maybe read a book, do some journaling, or meditate for a few minutes. 🧘‍♀️"],
    "neutral": ["Got it. 😊", "Alright! Let me know if you need anything. 💙", "Okay! I'm here whenever you need me."],
    "affirmation": [
        "You are worthy of love, happiness, and success. No matter what today brings, you are enough just as you are. 💙",
        "You have overcome challenges before, and you will overcome this too. Keep believing in yourself—you are capable of amazing things!",
        "You are strong, resilient, and capable. No obstacle is too big for you to handle. Trust in yourself and your journey. 🌟"
    ],
    "cheer_up": [
        "Of course! 💙 Remember, even on tough days, you are loved, valued, and capable of amazing things. Do you want a joke or a fun fact?",
        "Absolutely! You deserve happiness. Here’s a little reminder: The world is better with you in it. Now, should I tell you a joke or a cute animal fact? 🐶",
        "Hey, you’re doing better than you think! Life has its ups and downs, but you are stronger than you know. Want a silly joke or a heartwarming story?"
    ],
    "breathing_exercises": [
        "Got it. 😊 Let's do a quick breathing exercise. Breathe in... 1...2...3...4... Hold...1...2...3...4... Now slowly exhale...1...2...3...4... Feel any better? 💙",
        "Take a deep breath with me: Inhale slowly for 4 seconds... Hold it for 4 seconds... Now exhale slowly for 4 seconds... Repeat a few times. You got this! 🌿",
        "Try this: Close your eyes, take a deep breath in for 4 seconds, hold it for 4 seconds, and exhale for 4 seconds. Repeat until you feel calmer. 💙",
        "Let’s reset with a breathing exercise. Breathe in deeply… Hold it… And exhale slowly. Feel your body relaxing with each breath. I'm here with you. 💙"
    ],
    "positive_compliments":[
        "You're so sweet! 😊💙 I'm here for you.",
        "Aww, you're amazing! 😍💙",
        "You're too kind! Thank you for making my day better. 💙",
        "You're awesome! I'm lucky to chat with you! 💙",
        "You made me smile! You're such a ray of sunshine! 😊☀️"
    ],
    "refusal_replies":["I understand. It's okay to take your time. 💙",
            "That's perfectly fine. I'm here whenever you're ready. 💙",
            "No pressure, take all the time you need. 💙"
            ],
    "facts": ["Did you know that October 10th is World Mental Health Day? It's a day to raise awareness about mental health and promote positive mental well-being.","Studies show that laughter can reduce stress and improve your mood by releasing endorphins (the brain's 'feel-good' chemicals).",
              "Did you know that October 10th is World Mental Health Day? It's a day to raise awareness about mental health and promote positive mental well-being.",
                "Studies show that laughter can reduce stress and improve your mood by releasing endorphins (the brain's 'feel-good' chemicals).",
                "The color blue is often associated with calmness and relaxation. Some studies suggest that looking at the color blue can help reduce anxiety and promote a sense of peace.",
                "Did you know that regular physical activity doesn’t just improve your body? It also enhances your brain’s performance and can help reduce symptoms of depression and anxiety.",
                "Keeping a journal or writing down your thoughts and feelings can help you process emotions and reduce stress. This is why journaling is often recommended as a mental health exercise.",
                "Through a process called neuroplasticity, your brain has the ability to rewire itself, adapting to new situations and learning from experiences.",
                "Listening to music can have a positive effect on mental health by boosting your mood, reducing stress, and improving cognitive function. It's like a workout for your brain!",
                "The way you speak to yourself affects your emotional well-being. Positive self-talk can help reduce stress and improve confidence, while negative self-talk can contribute to feelings of anxiety.",
                "Sleep is crucial for your emotional well-being. It helps regulate mood, improve cognitive function, and reduce stress. So, getting 7-9 hours of quality sleep every night is key!",
                "Practicing gratitude has been shown to increase feelings of happiness and improve mental health. Taking a moment every day to appreciate the little things in life can make a big difference.",
                "Did you know that even if you're feeling down, smiling can trick your brain into thinking you're happy? This is known as the facial feedback hypothesis.",
                "Spending time with friends and family can improve mental health by reducing feelings of loneliness and increasing feelings of happiness and security.",
                "Mindfulness meditation can help lower anxiety, reduce stress, and improve focus. It's like a mini vacation for your brain!",
                "Spending time outdoors and connecting with nature has been shown to improve mood and reduce feelings of stress and anxiety. A walk in the park can make a big difference!",
                "Did you know that your brain physically reacts to your emotions? When you're stressed, your heart rate and blood pressure rise, and when you're calm, your body relaxes.","Believe it or not, laughing can burn up to 40 calories per hour! So the next time you're feeling down, put on a funny movie and laugh your stress away."
    ],
    "animalfacts" : [
    "Did you know that sea otters hold hands when they sleep to keep from drifting apart? It's called 'rafting,' and it helps them stay connected!",
    "A group of flamingos is called a 'flamboyance.' The name is fitting, considering their bright pink feathers and elegant poses!",
    "Kangaroos can’t walk backward! They are built for hopping and moving forward, making them excellent at jumping over obstacles.",
    "Elephants are known to have incredible memories. They can remember friends and places for many years, and they often show compassion and empathy towards each other.",
    "Quokkas, known as the 'happiest animals on Earth,' always seem to be smiling! They are small, herbivorous marsupials from Australia.",
    "Crows are highly intelligent and can even use tools. Some have been observed using sticks to fish for food, and they can even recognize human faces!",
    "A baby panda is born the size of a stick of butter! They grow rapidly over the first few months of life, gaining weight much faster than human babies.",
    "Sloths only defecate once a week! They do it at the base of a tree and often spend hours climbing down to do so.",
    "Penguins mate for life and often propose to their mates with a pebble. The male penguin will give a pebble to the female as a symbol of his love and commitment!",
    "Hedgehogs are known for their adorable 'hogs'! They curl up into a tight ball when scared, protecting their delicate underbellies with their spiky outer shells."
   ],
}

# Keyword categories
keywords = {
    "breathing_exercises": ["breathing exercise","breathe", "deep breath", "help me breathe", "calm down", "breathing"],
    "greeting": ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"],
    "motivation": ["motivate", "study", "focus", "inspire", "encourage"],
    "venting": ["i feel", "sad", "lonely", "depressed", "anxious", "lost", "vent"],
    "stress": ["stressed", "overwhelmed", "pressure", "tired", "burnt out"],
    "gratitude": ["thank you", "thanks", "grateful", "appreciate"],
    "casual": ["how are you", "what’s up", "what are you doing", "talk to me"],
    "joke": ["joke", "make me laugh", "funny"],
    "self_care": ["self-care", "self care", "well-being", "relax", "mental health", "take care"],
    "neutral": ["okay", "ok", "fine", "alright"],
    "affirmation": ["daily affirmation", "affirmation", "positive thoughts", "motivate me"],
    "cheer_up": ["cheer me up", "make me happy", "lift my mood"],
    "postive_compliments":["you are great","you're so good", "you're cute", "i love you","aww","you're the best","so sweet","you made my day"],
    "refusal":["i don't want to", "i do not want to"],
    "facts" : ["facts", "random facts"],
    "animals": ["animal facts","about animals"]
}

def categorize_input(user_input):
    user_input_lower = user_input.lower()
    joke_reactions = ["haha", "lol", "lmao", "that's funny", "so funny", "😂", "🤣"]
    if any(word in user_input_lower for word in joke_reactions):
        return "joke_reaction"
    
    for category, words in keywords.items():
        if any(word in user_input_lower for word in words):
            return category
    return "default"

def analyze_sentiment(user_input):
    inputs = tokenizer(user_input, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
    
    sentiment_score = torch.argmax(probs).item() + 1
    if sentiment_score <= 2:
        return "venting"
    elif sentiment_score == 3:
        return "stress"
    else:
        return "motivation"

def preprocess_input(user_input):
    # Expand contractions like "you're" to "you are"
    expanded_input = contractions.fix(user_input)
    return expanded_input


def generate_response(user_input):
    category = categorize_input(user_input)
    if category == "positive_compliments":
        return random.choice(responses["positive_compliments"])
    
    if category == "default":
        category = analyze_sentiment(user_input)
    
    # If the conversation is neutral or not emotionally charged, avoid repeating greetings
    if category == "neutral":
        return random.choice([
            "Got it! 😊 Let me know if you want to chat about anything.",
            "Alright! I'm here if you need anything. 💙",
            "Okay, feel free to share whenever you're ready. 💙"
        ])
    
    # Handle emotional responses if needed
    last_messages = [msg["text"] for msg in st.session_state.chat_history[-3:]] if "chat_history" in st.session_state else []
    emotional_categories = ["venting", "stress", "motivation", "cheer_up", "self_care", "affirmation"]

    # Avoid switching to a greeting if emotional conversation is ongoing
    if any(categorize_input(msg) in emotional_categories for msg in last_messages):
        if category == "greeting":
            category = "neutral"  # Don't greet when the conversation is emotional
    if 'cheer_up' in last_messages and "yes" in user_input.lower():
        return random.choice(responses["cheer_up"])  # Keep the cheer-up response going
    if 'fact' in last_messages and "yes" in user_input.lower():
        return random.choice(responses["facts"])  # Keep the cheer-up response going
    if 'motivation' in last_messages and "yes" in user_input.lower():
        return random.choice(responses["motivation"])  # Continue motivation if requested earlier
    if 'animal fact' in last_messages and "yes" in user_input.lower():
        return random.choice(responses["animalfacts"])
    if 'joke' in last_messages and "yes" in user_input.lower():
        return random.choice(responses["jokes"])  # Respond with a joke if they previously requested one
    if 'breathing' in last_messages and "yes" in user_input.lower():
        return "Great! Let’s do another round. Breathe in... 1...2...3...4... Hold...1...2...3...4... Now slowly exhale...1...2...3...4... How are you feeling now?"
    if 'talk about' in last_messages and "yes" in user_input.lower():
        return "what is it?"
    # Default behavior for 'yes' being a general affirmative response
    if "yes" in user_input.lower():
        return "Got it! Anything else you'd like to talk about?"
    # Handle responses for specific phrases indicating doubt or frustration
    if "i don’t think so" in user_input.lower() or "i can't" in user_input.lower() or "i'm not sure" in user_input.lower():
        return random.choice([
            "It’s okay to feel unsure sometimes. Remember, it's one step at a time. 💙",
            "I understand that it can be tough. You’ve handled tough moments before. I’m here with you. 💙",
            "Take it slow, you don't have to have everything figured out right now. I’m here for you. 💙"
        ])
    if category == "positive_compliments":
        return random.choice(responses["positive_compliments"])

    # If the input contains a joke reaction, handle it separately
    if category == "joke_reaction":
        return random.choice(["Haha, that's funny!", "Glad you liked that!", "I know, right?"])

    # Default response generation based on category
    return random.choice(responses.get(category, responses["neutral"]))


# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("💬 Chat with Me!")
st.write("I'm here to listen and support you. Click a prompt below or type your message.")

# Suggested prompts as buttons
suggested_prompts = [
    "I'm feeling stressed", 
    "I need motivation", 
    "Tell me a joke", 
    "I'm feeling lonely", 
    "Give me a self-care tip",
    "Help me deal with sadness", 
    "I'm feeling overwhelmed", 
    "Give me a daily affirmation", 
    "Can you cheer me up?"
]

cols = st.columns(min(len(suggested_prompts),3))
for i, prompt in enumerate(suggested_prompts):
    if cols[i % 3].button(prompt):
        st.session_state.chat_history.append({"role": "user", "text": prompt})
        bot_response = generate_response(prompt)
        st.session_state.chat_history.append({"role": "assistant", "text": bot_response})

# Chat input
user_input = st.chat_input("Type your message:", key="user_input")
if user_input:
    st.session_state.chat_history.append({"role": "user", "text": user_input})
    bot_response = generate_response(user_input)
    st.session_state.chat_history.append({"role": "assistant", "text": bot_response})

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["text"])
