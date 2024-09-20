import os
import random
import logging
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

#  Загрузка предобученной модели и токенизатора
model_name = 'sberbank-ai/rugpt2large'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Загрузка русскоязычных шуток из датасета
def load_jokes(filename: str):
    try:
        df = pd.read_csv(filename)
        jokes = df['text'].tolist()
        return jokes
    except Exception as e:
        logger.error(f"Ошибка при загрузке шуток из файла: {e}")
        return []

jokes_list = load_jokes('jokes.csv')

# Функция для генерации анекдотов
def generate_joke(prompt: str, max_length: int = 120):
    #prompt = 'Расскажи шутку, которая начинается вот так: ' + prompt
    input_ids = tokenizer.encode('Расскажи шутку, которая начинается вот так: ' + prompt, return_tensors='pt')

    # Настройки генерации
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=1.1,  # Высокая температура для более случайной генерации
            top_k=50,         
            top_p=0.95,      
            repetition_penalty=1.2  # Штраф за повторения
        )

    joke = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Обработка Промпта для модели
    joke = joke[len(prompt):].strip()

    #end_index = joke.find('.')
    #if end_index != -1:
    #    joke = joke[:end_index + 1]  # Обрезаем результат на оконченном предложении
    result = prompt + '\n' + joke.strip()
    return result

# Обработчик команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Привет! Я бот, который генерирует анекдоты. Напиши мне начало анекдота или просто 'шутка', чтобы получить случайный.")

# Обработчик любого другого
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text.strip()
    # Проверка на пустое сообщение
    if not text:
        await update.message.reply_text("Пожалуйста, напишите что-то, чтобы я смог сгенерировать анекдот.")
        return
        
    # Проверка на определенное начало анекдота или на шутку из датасета
    if text.lower() == 'шутка':
        if jokes_list:
            prompt = random.choice(jokes_list)
        else:
            await update.message.reply_text("Извините, шутки не загружены.")
            return
    else:
        prompt = text
    
    try:
        joke = generate_joke(prompt)
        if joke:  # Проверка на пустую выдачу
            await update.message.reply_text(joke)
        else:
            await update.message.reply_text("Извините, я не смог сгенерировать анекдот на основе этого запроса.")
    except Exception as e:
        logger.error(f"Ошибка при генерации анекдота: {e}")
        await update.message.reply_text("Извините, произошла ошибка при генерации анекдота.")

def main() -> None:
    token = "" # Здесь токен для бота, при публикации скрыл
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == '__main__':
    main()