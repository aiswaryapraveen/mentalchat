import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
import random
import contractions


model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


responses = {
    "greeting": [
        "Hey there! 😊 How are you feeling today?",
        "Hi! Hope you're doing well. 💙",
        "Hello! What’s on your mind?"
        ],
    "motivation": [
        "You’ve got this! Stay strong and keep going! 💪🚀",
        "Stay focused, break your tasks into small steps, and take it one step at a time! 📚",
        "Every step you take brings you closer to success. Keep pushing! 🔥"
        ],
    "venting": [
        "You deserve love no matter what head space you're in. If you don't have people to remind you of that when you need it, just know i am always here for each and every one of you.",
        "I'm here for you. You can share anything with me. 💙", 
        "It's okay to feel this way. You're not alone. I'm here to listen. 💙",
        "Take a deep breath. I'm here for you. Let's talk."
        ],
    "stress": [
        "Life is overwhelming. You might be in school, you might have a job, or even just the demanding expectations in everyday life can cause crazy amounts of stress, everyone needs a break every now and then. You're doing great! im so proud of you",
        "Deep breaths. You’ve handled tough situations before, and you’ll get through this too. 💙",
        "Try taking a short break or going for a walk. You’re doing great. 🌿",
        "Remember to take things one step at a time. You’re stronger than you think!",
        "Life is crazy sometimes, you're not being dramatic,you're doing just fine! Take a few minutes if you can & just breathe, take some time to ground yourself and be aware of your surroundings."
        ],
    "gratitude": [
        "That’s so kind of you! I appreciate you too! 😊",
        "You're awesome! Thank you for sharing positivity. 💙",
        "Your kindness makes the world brighter. Thank you! 😊"
        ],
    "casual": [
        "I’m doing great! Thanks for asking. How about you? 😊",
        "I'm here and ready to chat! What's on your mind? 💙",
        "Feeling good! How’s your day going? 🌞",
        ],
    "joke": [
        "Why don’t skeletons fight each other? They don’t have the guts! 😆",
        "What do you call fake spaghetti? An impasta! 🍝😂",
        "Why did the scarecrow win an award? Because he was outstanding in his field! 🌾🤣"
        ],
    "self_care": [
        "Taking care of yourself is important! Try deep breathing, listening to music, or taking a short walk. 🌿",
        "Self-care tip: Stay hydrated, take breaks, and do something you love today! 💙",
        "A little self-care goes a long way! Maybe read a book, do some journaling, or meditate for a few minutes. 🧘‍♀️"
        ],
    "neutral": [
        "Got it. 😊", "Alright! Let me know if you need anything. 💙",
        "Okay! I'm here whenever you need me.","I'm here for you. 💙",
        "You're not alone. Take your time, and I'm listening. 😊",
        "I hear you. If you need to talk, I'm here for you. 💙"
        ],
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
    "refusal_replies":[
            "I understand. It's okay to take your time. 💙",
            "That's perfectly fine. I'm here whenever you're ready. 💙",
            "No pressure, take all the time you need. 💙",
            "You're not alone. Take your time, and I'm listening. 😊",
            "I hear you. If you need to talk, I'm here for you. 💙"
        ],
    "facts": [
        "Did you know that October 10th is World Mental Health Day? It's a day to raise awareness about mental health and promote positive mental well-being.","Studies show that laughter can reduce stress and improve your mood by releasing endorphins (the brain's 'feel-good' chemicals).",
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
   "stories":[
       "The Gift of a Lifetime: A young girl who had always wanted a puppy finally receives one on her birthday. She names him 'Hope' because of the joy he brought into her life. Through tough times, the bond between them strengthens, showing how love and companionship can heal.",
        "The Kindness of Strangers: After losing her job and struggling to make ends meet, a single mother was anonymously gifted enough groceries to feed her family for months. Later, she learned it was her former co-workers who had quietly rallied to help.","A Letter from Dad: A soldier stationed overseas writes heartfelt letters to his young daughter, telling her stories of how much he misses her and how proud he is of her. When he returns home, she has an album of all the letters ready, filled with love and memories.","The Rescued Dog: A dog was abandoned at a shelter, fearful and traumatized. After months of care and patience, the dog found a new home with a family that loved him unconditionally. Years later, he became the family’s loyal protector, always by their side.",
        "The Last-Minute Donation: A man who was struggling with financial difficulties noticed a woman outside a store holding a sign for help. He gave her all the cash in his wallet, only to later discover that she was the one who had been trying to raise funds for a local charity he supported.",
        "The Surprise Reunion: After years of being separated, two childhood friends reconnected on social media. One had been adopted by a new family after her parents’ passing. The reunion was emotional and heartwarming, and they picked up right where they left off, realizing how much they meant to each other.",
        "The Gift of Learning: A retired teacher volunteered to tutor a struggling student for free, despite having no formal teaching position. With her help, the student went from failing to passing with flying colors and was able to fulfill his dream of going to college.",
        "The Power of a Smile: A man having the worst day of his life entered a coffee shop and was greeted by a barista with a genuine smile. That small act of kindness completely turned his day around and gave him the courage to face his challenges with a renewed spirit.",
        "The Birthday Surprise: A young boy whose family couldn't afford birthday presents received a surprise package from an anonymous donor. Inside were toys, clothes, and a heartfelt birthday card. The family was deeply touched by the generosity of a stranger.","The Heartfelt Apology: A man who had distanced himself from his best friend years ago due to a misunderstanding sent an apology letter. To his surprise, his friend responded with forgiveness and shared how much he had missed their friendship. They rekindled their bond, proving it’s never too late to make amends."
        ],
    "thinking": [
        "That sounds interesting! What have you been thinking about?",
        "I’d love to hear more about that!",
        "It’s good to take a moment and reflect. What’s on your mind?",
        "Take your time—thoughts can be powerful.",
        "Sometimes just thinking things through can help bring clarity.",
        ],
    "positive_feedback" : [
        "I'm glad you liked it! 😊 Want to hear more jokes, facts, or stories?",
        "That's awesome! 😄 Would you like to hear another joke, fact, or maybe a story?",
        "I love that you're enjoying this! Do you want to hear more? 😊"
    ],
    "goodbye" : [
        "Goodbye! Take care! 💙",
        "See you soon! Don't hesitate to come back! 😊",
        "Take care! I'll be here if you need me. 💙",
    ],
    "support" : [
        "I'm here for you! What do you need help with?",
        "Let’s figure this out together. What’s on your mind?",
        "I’m all ears! Tell me how I can assist."   
    ],
    "small_talk" :[
        "Just here to chat with you! 😊 What about you?",
        "Not much, just waiting to have a nice conversation. What's on your mind?",
        "I'm here to listen! What’s up?",
    ],

    }


keywords = {
    "breathing_exercises": ["breathing exercise","breathe", "deep breath", "help me breathe", "calm down", "breathing"],
    "greeting": ["bro","hi","hai","hellaur", "hello", "hey", "good morning", "good afternoon", "good evening"],
    "goodbye":[ "bye", "goodbye" ,"see you later" ,"take care", "see ya"],
    "motivation": ["motivate", "study", "focus", "inspire", "encourage"],
    "venting": ["i feel", "sad", "lonely", "depressed", "anxious", "lost", "vent"],
    "stress": ["stressed", "overwhelmed", "pressure", "tired", "burnt out"],
    "gratitude": ["thank you", "thanks", "grateful", "appreciate"],
    "casual": ["how are you", "what is up","how are you doing","how do you do"],
    "joke": ["joke", "make me laugh", "funny"],
    "self_care": ["self-care", "self care", "well-being", "relax", "mental health", "take care"],
    "neutral": ["okay", "ok", "fine", "alright"],
    "affirmation": ["daily affirmation", "affirmation", "positive thoughts", "motivate me"],
    "cheer_up": ["cheer me up", "make me happy", "lift my mood"],
    "positive_compliments":["you are great","you are so good", "you are cute", "i love you","aww","you are the best","so sweet","you made my day"],
    "refusal":["i don't want to", "i do not want to","nothing", "no", "i do not"],
    "facts" : ["facts", "random facts", "fact"],
    "animals": ["animal facts","about animals"],
    "story":["story","tell me a story","i would like a story"],
    "positive_feedback":["that is so good","cool","oh damn","that is cool","that is good", "that is great","awesome", "i love that", "so good", "amazing", "awesome", "so cool", "so cute", "cute"],
    "support" : ["help", "support", "need advice", "how can I help" ,"stuck"],
    "thinking": ["thinking", "pondering","deep thought","reflecting"],
    "small_talk":["what are you doing", "how is your day", "what is up"]
}
overwhelmed_responses = [
    "Life can feel like a lot sometimes, and that's okay. You're doing your best, and that's enough. Take a deep breath and give yourself grace. 💙",
    "You're not alone in this. It’s okay to feel overwhelmed. Try focusing on one thing at a time, and remember, you are stronger than you think! 💪",
    "I hear you. It sounds like a lot is on your mind. Maybe taking a short break, listening to some music, or journaling your thoughts might help. ✨",
    "You’re not being dramatic, you're human. Take a deep breath and allow yourself to pause for a moment. You’ve got this. 💙",
    "Feeling overwhelmed is tough, but you don’t have to go through it alone. If you want to share, I’m here to listen. No pressure. 🤗"
]

follow_up_responses = [
    "Okay, take your time. I'm here whenever you're ready to talk. 💙",
    "No rush at all. Just know that I’m here for you. 🤗",
    "Whenever you're ready, I'm here to listen. You're not alone. 💜",
    "It's okay. You don’t have to have all the answers right now. One step at a time. 🌿",
    "Take your time. If and when you're ready to share, I’ll be right here. 🫂"
]

def categorize_input(user_input):
    user_input_lower = user_input.lower()

    priority_categories = ["venting", "stress", "thinking", "support"]
    
    # Check priority categories first
    for cat in priority_categories:
        if any(word in user_input for word in keywords[cat]):
            return cat
    words = user_input.lower().split()  # Split into words
    
    for cat, word_list in keywords.items():
        if any(word in words for word in word_list):  # Match exact words
            return cat

    refusal_words = ["nothing", "no", "i don’t", "i do not", "i dont", "not really", "not now"]
    if any(word in user_input_lower for word in refusal_words):
        return "refusal_replies"
    
    joke_reactions = ["haha", "lol", "lmao", "that's funny", "so funny", "😂", "🤣"]
    if any(word in user_input_lower for word in joke_reactions):
        return "joke_reaction"
    
    animals = ["animal facts","about animals", "animal fact"]
    if any(word in user_input_lower for word in animals):
        return "animalfacts"
    
    facts =["fact", "random fact", "facts"]
    if any(word in user_input_lower for word in facts):
        return "facts"
    
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
    
    if sentiment_score == 1:
        return "cheer_up"  # Strongly negative (-1.0 to -0.7)
    elif sentiment_score == 2:
        return "affirmation"  # Mildly negative (-0.69 to -0.3)
    elif sentiment_score == 3:
        return "neutral"  # Neutral (-0.29 to 0.29)
    elif sentiment_score == 4:
        return "self_care"  # Mildly positive (0.3 to 0.69)
    else:  # sentiment_score == 5
        return "motivation"  # Strongly positive (0.7 to 1.0)

def preprocess_input(user_input):
    expanded_input = contractions.fix(user_input)
    return expanded_input

def generate_response(user_input):
    user_input = preprocess_input(user_input)
    category = categorize_input(user_input)
    print(f"Category identified: {category}")
    if category == "default":
        category = analyze_sentiment(user_input)

    last_messages = [msg["text"] for msg in st.session_state.chat_history[-3:]] if "chat_history" in st.session_state else []
    last_messages_categories = [categorize_input(msg) for msg in last_messages]
    print(last_messages)
    print(last_messages_categories)
    if category == "positive_feedback":
        response = random.choice(responses["positive_feedback"])
        if any(phrase in response.lower() for phrase in ["joke","fact","animal fact", "story"]):
            st.session_state.follow_up ="positive_feedback"
        return response
    
    if category == "cheer_up":
        response = random.choice(responses["cheer_up"])
        
        # If the response includes a follow-up question, track it
        if any(phrase in response.lower() for phrase in ["joke", "fact", "animal fact", "stories"]):
            st.session_state.follow_up = "cheer_up" 
        
        return response
    if "yes" in user_input.lower():
        if st.session_state.get("follow_up") == "positive_feedback":
            st.session_state.follow_up = "joke_fact_story"  # Reset flag after responding
            return random.choice(responses[random.choice(["joke", "facts", "animalfacts", "stories"])])
        
        if st.session_state.get("follow_up") == "joke_fact_story":
            return random.choice(responses[random.choice(["joke","facts","animalfacts","stories"])])
        
        if st.session_state.get("follow_up") == "cheer_up":
            st.session_state.follow_up = None  # Reset flag after responding
            return random.choice(responses[random.choice(["joke", "facts", "animalfacts", "stories"])])
        if "story" in last_messages_categories:
            return random.choice(responses["stories"])
        if "facts" in last_messages_categories:
            return random.choice(responses["facts"])
        if "motivation" in last_messages_categories:
            return random.choice(responses["motivation"])
        if "animalfacts" in last_messages_categories:
            return random.choice(responses["animalfacts"])
        if "jokes" in last_messages_categories:
            return random.choice(responses["jokes"])
        if "breathing" in last_messages_categories:
            return "Great! Let’s do another round. Breathe in... 1...2...3...4... Hold...1...2...3...4... Now slowly exhale...1...2...3...4... How are you feeling now?"
        return "Got it! Anything else you'd like to talk about?"
    
     # Direct story and fact responses
    if "story" in user_input.lower():
        return random.choice(responses["stories"])
    if "animal facts" in user_input.lower():
        return random.choice(responses["animalfacts"])
    
    if 'last_story_given' in st.session_state and st.session_state.last_story_given:
        # If a story was provided recently, avoid giving a motivational message unless prompted
        if "motivate" in user_input.lower() or "cheer up" in user_input.lower():
            return random.choice(responses["motivation"])
        if user_input.strip() == "":
            return "Alright! Let me know if you need anything. 💙"
    
    if 'last_animal_fact_given' in st.session_state and st.session_state.last_animal_fact_given:
        if user_input.strip() == "":
            return "Alright! Let me know if you need anything. 💙"
    
    if category == "stress" or category == "venting":
        return random.choice(responses[category])  # Ensure stress gets appropriate responses
    
    if category == "neutral":   
        return random.choice([
            "Got it! 😊 Let me know if you want to chat about anything.",
            "Alright! I'm here if you need anything. 💙",
            "Okay, feel free to share whenever you're ready. 💙"
        ])
    last_messages = [msg["text"] for msg in st.session_state.chat_history[-3:]] if "chat_history" in st.session_state else []
    emotional_categories = ["venting", "stress", "thinking","motivation", "cheer_up", "self_care", "affirmation"]

    if any(categorize_input(msg) in emotional_categories for msg in last_messages):
        if category == "greeting":
            category = "neutral"
    if "i don’t think so" in user_input.lower() or "i can't" in user_input.lower() or "i'm not sure" in user_input.lower():
        return random.choice([
            "It’s okay to feel unsure sometimes. Remember, it's one step at a time. 💙",
            "I understand that it can be tough. You’ve handled tough moments before. I’m here with you. 💙",
            "Take it slow, you don't have to have everything figured out right now. I’m here for you. 💙"
        ])
    
    if category == "joke_reaction":
        return random.choice(["Haha, that's funny!", "Glad you liked that!", "I know, right?"])

    return random.choice(responses.get(category, responses["neutral"]))

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("💬 Chat with Me!")
st.write("I'm here to listen and support you. Click a prompt below or type your message.")

suggested_prompts = [
    "I'm stressed", 
    "I need motivation", 
    "Tell me a joke",
    "Tell me a heartwarming story", 
    "I'm feeling lonely", 
    "Give me a self-care tip",
    "Help me deal with sadness", 
    "I'm feeling overwhelmed", 
    "Can you cheer me up?"
]

cols = st.columns(min(len(suggested_prompts),3))
for i, prompt in enumerate(suggested_prompts):
    if cols[i % 3].button(prompt):
        st.session_state.chat_history.append({"role": "user", "text": prompt})
        bot_response = generate_response(prompt)
        st.session_state.chat_history.append({"role": "assistant", "text": bot_response})


user_input = st.chat_input("Type your message:", key="user_input")
if user_input:
    st.session_state.chat_history.append({"role": "user", "text": user_input})
    bot_response = generate_response(user_input)
    st.session_state.chat_history.append({"role": "assistant", "text": bot_response})


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["text"])
