from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from datetime import datetime
import os
import json
import base64
import io
import numpy as np
from PIL import Image
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
import tempfile
import traceback
import tensorflow as tf
from tensorflow.keras.models import load_model
import uuid 

app = Flask(__name__, template_folder='templates')
CORS(app)

app.config['SECRET_KEY'] = 'health-ai-scanner-secret-key-2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# API ключ OpenAI (замените на ваш или используйте переменные окружения)
client = OpenAI(api_key='') # используйте ваш новый ключ
MODELS_CONFIG = {
    'xray': {
        'file': 'xray_disease_classifier_final.keras',
        'classes': ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
                    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
                    'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'],
        'grayscale': True
    },
    'nails': {
        'file': 'nail_disease_classifier_final.keras',
        'classes': ['Blue_finger', 'Clubbing', 'Healthy', 'Melanoma', 'Onychogryphosis', 'Pitting'],
        'grayscale': False
    },
    'skin': {
        'file': 'skin_disease_classifier_final.keras',
        'classes': ['Carcinoma', 'Dermatitis', 'Eczema', 'Fungi', 'Keratoses', 
                    'Keratosis', 'Melanoma', 'Nevi', 'Psoriasis', 'Warts'],
        'grayscale': False
    },
    'eyes': {
        'file': 'eye_disease_classifier_final.keras',
        'classes': ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal'],
        'grayscale': False
    }
}

DISEASES_DATABASE = [
    {
        "id": 5,
        "name": "Гипертония",
        "category": "Сердце и сосуды",
        "description": "Стойкое повышение артериального давления. Повышает риск инсультов и инфарктов.",
        "specialists": ["Кардиолог", "Терапевт"],
        "symptoms": ["Головная боль", "Головокружение", "Шум в ушах"],
        "recommendations": ["Ограничение соли", "Контроль давления", "Отказ от курения"]
    },
    {
        "id": 6,
        "name": "Сахарный диабет 2 типа",
        "category": "Эндокринная система",
        "description": "Нарушение обмена глюкозы из-за снижения чувствительности клеток к инсулину.",
        "specialists": ["Эндокринолог"],
        "symptoms": ["Постоянная жажда", "Частое мочеиспускание", "Утомляемость"],
        "recommendations": ["Диета с низким ГИ", "Физическая активность", "Контроль сахара"]
    },
    {
        "id": 7,
        "name": "Гастрит",
        "category": "ЖКТ",
        "description": "Воспаление слизистой оболочки желудка, часто связанное с бактерией H. pylori или питанием.",
        "specialists": ["Гастроэнтеролог"],
        "symptoms": ["Боль в эпигастрии", "Изжога", "Тошнота"],
        "recommendations": ["Дробное питание", "Отказ от острого", "Лечение по схеме врача"]
    },
    {
        "id": 8,
        "name": "Мигрень",
        "category": "Неврология",
        "description": "Первичная форма головной боли, проявляющаяся сильными приступами, чаще с одной стороны.",
        "specialists": ["Невролог"],
        "symptoms": ["Пульсирующая боль", "Светобоязнь", "Тошнота"],
        "recommendations": ["Соблюдение режима сна", "Избегание триггеров", "Прием специфических препаратов"]
    },
    {
        "id": 9,
        "name": "Астма",
        "category": "Легкие",
        "description": "Хроническое воспаление дыхательных путей, вызывающее приступы удушья.",
        "specialists": ["Пульмонолог", "Аллерголог"],
        "symptoms": ["Свистящее дыхание", "Одышка", "Чувство стеснения в груди"],
        "recommendations": ["Использование ингалятора", "Дыхательная гимнастика", "Устранение аллергенов"]
    },
    {
        "id": 10,
        "name": "Анемия",
        "category": "Кровь",
        "description": "Снижение уровня гемоглобина и/или эритроцитов в крови.",
        "specialists": ["Гематолог", "Терапевт"],
        "symptoms": ["Бледность кожи", "Слабость", "Ломкость волос"],
        "recommendations": ["Продукты с железом", "Прием витамина B12", "Анализ крови"]
    },
    {
        "id": 11,
        "name": "Остеохондроз",
        "category": "Опорно-двигательный аппарат",
        "description": "Дегенеративные изменения в межпозвоночных дисках.",
        "specialists": ["Невролог", "Вертебролог"],
        "symptoms": ["Боль в спине", "Онемение конечностей", "Ограничение подвижности"],
        "recommendations": ["ЛФК", "Массаж", "Правильная осанка"]
    },
    {
        "id": 12,
        "name": "Цистит",
        "category": "Мочевыделительная система",
        "description": "Воспаление мочевого пузыря, чаще бактериального характера.",
        "specialists": ["Уролог"],
        "symptoms": ["Болезненное мочеиспускание", "Частые позывы", "Мутная моча"],
        "recommendations": ["Тепло на низ живота", "Увеличение потребления воды", "Антибиотикотерапия"]
    },
    {
        "id": 13,
        "name": "Глаукома",
        "category": "Глаза",
        "description": "Заболевание, связанное с повышением внутриглазного давления и поражением зрительного нерва.",
        "specialists": ["Офтальмолог"],
        "symptoms": ["Радужные круги перед глазами", "Сужение полей зрения", "Резкая боль в глазу"],
        "recommendations": ["Глазные капли", "Контроль ВГД", "Хирургическое вмешательство"]
    },
    {
        "id": 14,
        "name": "Псориаз",
        "category": "Кожа",
        "description": "Хроническое неинфекционное заболевание, поражающее кожу и иногда суставы.",
        "specialists": ["Дерматолог"],
        "symptoms": ["Красные бляшки", "Шелушение", "Зуд"],
        "recommendations": ["Увлажнение кожи", "Фототерапия", "Снижение уровня стресса"]
    },
    {
        "id": 15,
        "name": "Гипотиреоз",
        "category": "Эндокринная система",
        "description": "Состояние, вызванное длительным недостатком гормонов щитовидной железы.",
        "specialists": ["Эндокринолог"],
        "symptoms": ["Отечность", "Замедление пульса", "Сонливость"],
        "recommendations": ["Гормональная терапия", "Йодированная соль", "Контроль ТТГ"]
    },
    {
        "id": 16,
        "name": "Аритмия",
        "category": "Сердце и сосуды",
        "description": "Нарушение частоты, ритмичности и последовательности сокращений сердца.",
        "specialists": ["Кардиолог", "Аритмолог"],
        "symptoms": ["Замирание сердца", "Учащенное сердцебиение", "Слабость"],
        "recommendations": ["ЭКГ-мониторинг", "Ограничение кофеина", "Прием антиаритмиков"]
    },
    {
        "id": 17,
        "name": "Холецистит",
        "category": "ЖКТ",
        "description": "Воспаление желчного пузыря, часто на фоне желчнокаменной болезни.",
        "specialists": ["Гастроэнтеролог", "Хирург"],
        "symptoms": ["Боль в правом подреберье", "Горечь во рту", "Желтушность"],
        "recommendations": ["Диета №5", "УЗИ брюшной полости", "Дробное питание"]
    },
    {
        "id": 18,
        "name": "Артрит",
        "category": "Суставы",
        "description": "Воспалительное заболевание суставов различной этиологии.",
        "specialists": ["Ревматолог"],
        "symptoms": ["Опухание суставов", "Скованность по утрам", "Локальное повышение температуры"],
        "recommendations": ["Умеренные нагрузки", "Противовоспалительные средства", "Физиотерапия"]
    },
    {
        "id": 19,
        "name": "Отит",
        "category": "ЛОР-органы",
        "description": "Воспалительный процесс в различных отделах уха.",
        "specialists": ["Оториноларинголог"],
        "symptoms": ["Боль в ухе", "Снижение слуха", "Выделения из ушного канала"],
        "recommendations": ["Защита ушей от воды", "Ушные капли", "Консультация врача"]
    },
    {
        "id": 20,
        "name": "Стоматит",
        "category": "Полость рта",
        "description": "Поражение слизистой оболочки полости рта с образованием язвочек.",
        "specialists": ["Стоматолог", "Терапевт"],
        "symptoms": ["Язвы во рту", "Болезненность при еде", "Повышенное слюноотделение"],
        "recommendations": ["Антисептические полоскания", "Мягкая пища", "Гигиена рта"]
    },
    {
        "id": 21,
        "name": "Варикоз",
        "category": "Сердце и сосуды",
        "description": "Расширение поверхностных вен, сопровождающееся нарушением кровотока.",
        "specialists": ["Флеболог", "Хирург"],
        "symptoms": ["Тяжесть в ногах", "Сосудистые звездочки", "Отеки к вечеру"],
        "recommendations": ["Компрессионный трикотаж", "Контрастный душ", "Пешие прогулки"]
    },
    {
        "id": 22,
        "name": "Депрессия",
        "category": "Психика",
        "description": "Психическое расстройство, характеризующееся снижением настроения и утратой интереса к жизни.",
        "specialists": ["Психиатр", "Психотерапевт"],
        "symptoms": ["Апатия", "Нарушение сна", "Чувство вины"],
        "recommendations": ["Психотерапия", "Режим дня", "Медикаментозная поддержка по назначению"]
    },
    {
        "id": 23,
        "name": "Тонзиллит (Ангина)",
        "category": "ЛОР-органы",
        "description": "Инфекционное воспаление небных миндалин.",
        "specialists": ["Оториноларинголог"],
        "symptoms": ["Боль при глотании", "Налет на миндалинах", "Увеличение лимфоузлов"],
        "recommendations": ["Полоскание горла", "Щадящая диета", "Соблюдение курса антибиотиков"]
    },
    {
        "id": 24,
        "name": "Гайморит",
        "category": "ЛОР-органы",
        "description": "Воспаление слизистой оболочки гайморовой пазухи.",
        "specialists": ["Оториноларинголог"],
        "symptoms": ["Заложенность носа", "Давление в области скул", "Головная боль"],
        "recommendations": ["Промывание носа", "Сосудосуживающие капли", "Рентген пазух"]
    },
    {
        "id": 25,
        "name": "Аллергический ринит",
        "category": "Аллергология",
        "description": "Воспаление слизистой носа, вызванное аллергенами.",
        "specialists": ["Аллерголог", "ЛОР"],
        "symptoms": ["Чихание", "Зуд в носу", "Прозрачные выделения"],
        "recommendations": ["Исключение контакта с аллергеном", "Антигистаминные", "Влажная уборка"]
    },
    {
        "id": 26,
        "name": "Экзема",
        "category": "Кожа",
        "description": "Хроническое воспаление кожи аллергической природы.",
        "specialists": ["Дерматолог", "Аллерголог"],
        "symptoms": ["Высыпания с пузырьками", "Сильный зуд", "Мокнутие кожи"],
        "recommendations": ["Диета", "Защитные кремы", "Избегание агрессивной химии"]
    },
    {
        "id": 27,
        "name": "Пиелонефрит",
        "category": "Мочевыделительная система",
        "description": "Инфекционное воспаление почек.",
        "specialists": ["Нефролог", "Уролог"],
        "symptoms": ["Боль в пояснице", "Высокая температура", "Озноб"],
        "recommendations": ["Постельный режим", "Диета с ограничением соли", "Анализ мочи по Нечипоренко"]
    },
    {
        "id": 28,
        "name": "Радикулит",
        "category": "Неврология",
        "description": "Воспаление или защемление корешков спинномозговых нервов.",
        "specialists": ["Невролог"],
        "symptoms": ["Простреливающая боль", "Снижение чувствительности", "Слабость мышц"],
        "recommendations": ["Сухое тепло", "Покой", "Прием анальгетиков"]
    },
    {
        "id": 29,
        "name": "Панкреатит",
        "category": "ЖКТ",
        "description": "Воспаление поджелудочной железы.",
        "specialists": ["Гастроэнтеролог"],
        "symptoms": ["Опоясывающая боль", "Рвота", "Нарушение стула"],
        "recommendations": ["Голод, холод и покой в острый период", "Ферментные препараты", "Отказ от алкоголя"]
    }
]

DISEASE_CLASSES = {
    'xray': [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
        'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
        'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
    ],
    'nails': [
        'Blue_finger', 'Clubbing', 'Healthy', 'Melanoma', 'Onychogryphosis', 'Pitting'
    ],
    'skin': [
        'Carcinoma', 'Dermatitis', 'Eczema', 'Fungi', 'Keratoses', 
        'Keratosis', 'Melanoma', 'Nevi', 'Psoriasis', 'Warts'
    ],
    'eyes': [
        'cataract', 'diabetic_retinopathy', 'glaucoma', 'normal'
    ]
}

# Словарь для перевода и поиска в базе данных (DISEASE_DETAILS)
TRANSLATIONS = {
    # X-ray
    'Atelectasis': 'Ателектаз', 'Cardiomegaly': 'Кардиомегалия', 'Consolidation': 'Консолидация',
    'Edema': 'Отек легких', 'Effusion': 'Плевральный выпот', 'Emphysema': 'Эмфизема',
    'Fibrosis': 'Фиброз', 'Hernia': 'Грыжа', 'Infiltration': 'Инфильтрация',
    'Mass': 'Новообразование', 'No Finding': 'Норма (Рентген)', 'Nodule': 'Узелок',
    'Pleural_Thickening': 'Утолщение плевры', 'Pneumonia': 'Пневмония', 'Pneumothorax': 'Пневмоторакс',
    # Ногти
    'Blue_finger': 'Цианоз ногтей', 'Clubbing': 'Пальцы Гиппократа', 'Healthy': 'Здоровые ногти',
    'Onychogryphosis': 'Онихогрифоз', 'Pitting': 'Наперстковидная истыканность',
    # Лицо/Кожа
    'Carcinoma': 'Карцинома', 'Dermatitis': 'Дерматит', 'Eczema': 'Экзема',
    'Fungi': 'Грибок', 'Keratoses': 'Кератоз', 'Keratosis': 'Кератоз',
    'Melanoma': 'Меланома', 'Nevi': 'Невус (Родинка)', 'Psoriasis': 'Псориаз', 'Warts': 'Бородавки',
    # Глаза
    'cataract': 'Катаракта', 'diabetic_retinopathy': 'Ретинопатия', 
    'glaucoma': 'Глаукома', 'normal': 'Здоровые глаза'
}

def load_model_safe(model_path):
    try:
        if os.path.exists(model_path):
            print(f"Загрузка модели: {model_path}...")
            return load_model(model_path)
        print(f"Файл модели {model_path} не найден!")
        return None
    except Exception as e:
        print(f"Ошибка при загрузке {model_path}: {e}")
        return None

# Загружаем новые модели .keras
eyes_model = load_model_safe('eye_disease_classifier_final.keras')
nails_model = load_model_safe('nail_disease_classifier_final.keras')
xray_model = load_model_safe('xray_disease_classifier_final.keras')
skin_model = load_model_safe('face_disease_classifier_final.keras')

def preprocess_image(image_data, target_size=(224, 224), grayscale=False):
    """Подготовка изображения с учетом цветности"""
    if "," in image_data:
        image_data = image_data.split(",")[1]
    
    img_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(img_bytes))
    
    if grayscale:
        # Конвертируем в 1 канал (Grayscale)
        img = img.convert('L')
    else:
        # Конвертируем в 3 канала (RGB)
        img = img.convert('RGB')
        
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    
    # Добавляем размерность батча
    img_array = np.expand_dims(img_array, axis=0)
    
    # Если это ч/б, нужно добавить размерность канала в конце: (1, 224, 224, 1)
    if grayscale:
        img_array = np.expand_dims(img_array, axis=-1)
        
    return img_array

def predict_disease(model, img_array, category):
    """Универсальная функция предсказания для многоклассовых моделей"""
    if model is None:
        return {"disease": "Модель не загружена", "confidence": 0.0, "is_normal": True}
    
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][class_idx])
    
    raw_name = DISEASE_CLASSES[category][class_idx]
    display_name = TRANSLATIONS.get(raw_name, raw_name)
    
    # Определяем, является ли результат "здоровым"
    normal_keywords = ['normal', 'Healthy', 'No Finding']
    is_normal = raw_name in normal_keywords
    
    return {
        "disease": display_name,
        "confidence": round(confidence * 100, 1),
        "is_normal": is_normal,
        "raw_name": raw_name
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/scan', methods=['POST'])
def scan_image():
    try:
        data = request.json
        image_data = data.get('image')
        scan_type = data.get('type') # 'skin', 'eyes', 'nails', 'xray'

        if not image_data or not scan_type:
            return jsonify({"error": "Недостаточно данных"}), 400

        img_array = preprocess_image(image_data)
        
        # Выбор модели
        models_map = {
            'skin': (skin_model, 'skin'),
            'eyes': (eyes_model, 'eyes'),
            'nails': (nails_model, 'nails'),
            'xray': (xray_model, 'xray')
        }
        
        current_model, cat_name = models_map.get(scan_type)
        result = predict_disease(current_model, img_array, cat_name)
        
        return jsonify({
            "status": "success",
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        # Получаем текст сообщения из JS (поле 'message')
        user_text = data.get('message', '')
        # Получаем историю (если есть)
        history = data.get('history', [])

        if not user_text:
            return jsonify({"success": False, "error": "Пустое сообщение"}), 400

        # Формируем сообщения для OpenAI
        messages_for_ai = [
            {"role": "system", "content": "Вы — медицинский ИИ-ассистент сервиса ainala.ai. Отвечайте кратко и профессионально."}
        ]
        
        # Добавляем историю для контекста
        for h in history:
            role = "assistant" if h.get('sender') == 'ai' else "user"
            messages_for_ai.append({"role": role, "content": h.get('text', '')})
            
        # Добавляем текущее сообщение
        messages_for_ai.append({"role": "user", "content": user_text})

        # Запрос к OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages_for_ai
        )
        
        ai_reply = response.choices[0].message.content

        # ВАЖНО: Возвращаем формат, который ждет твой JS
        return jsonify({
            "success": True,
            "message": ai_reply
        })

    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({
            "success": False, 
            "error": str(e)
        }), 500

@app.route('/api/analyze-xray', methods=['POST'])
def analyze_xray():
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({"success": False, "error": "Изображение не получено"}), 400
            
        # Подготовка изображения (Grayscale для рентгена)
        img_array = preprocess_image(image_data, grayscale=True)
        
        # Предсказание
        predictions = xray_model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        categories = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", 
            "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", 
            "No Finding", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
        ]
        
        disease_name = categories[class_idx]
        
        # Словарь переводов (чтобы в интерфейсе было красиво)
        translations = {
            "No Finding": "Норма", "Pneumonia": "Пневмония", "Edema": "Отек легких",
            "Cardiomegaly": "Кардиомегалия", "Effusion": "Плевральный выпот"
            # добавьте остальные по желанию
        }
        
        ru_name = translations.get(disease_name, disease_name)

        # Формируем ответ, который ждет ваш JS (data.success и data.result)
        return jsonify({
            "success": True,  # Именнно 'success', а не 'status'
            "analysis_id": str(uuid.uuid4()), # Генерируем ID для переменной currentAnalysisId
            "result": {
                "disease": ru_name,
                "name": ru_name, # На случай если JS ищет .name
                "confidence": round(confidence * 100, 1),
                "is_normal": disease_name == "No Finding",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False, 
            "error": f"Ошибка нейросети: {str(e)}"
        }), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_general():
    try:
        data = request.json
        image_data = data.get('image')
        scan_type = data.get('type') # 'skin', 'eyes', 'nails'
        
        if scan_type not in MODELS_CONFIG:
            return jsonify({"success": False, "error": "Неверный тип анализа"}), 400

        config = MODELS_CONFIG[scan_type]
        # Используем grayscale=False для кожи, глаз и ногтей
        img_array = preprocess_image(image_data, grayscale=config['grayscale'])
        
        # Получаем модель из глобального словаря загруженных моделей
        # (Убедитесь, что вы загрузили их при старте сервера как skin_model, nails_model и т.д.)
        target_model = None
        if scan_type == 'skin': target_model = skin_model
        if scan_type == 'eyes': target_model = eyes_model
        if scan_type == 'nails': target_model = nails_model

        preds = target_model.predict(img_array)
        idx = np.argmax(preds[0])
        disease_name = config['classes'][idx]
        
        return jsonify({
            "success": True,
            "analysis_id": str(uuid.uuid4()),
            "result": {
                "disease": disease_name,
                "confidence": round(float(preds[0][idx]) * 100, 1),
                "is_normal": disease_name in ['normal', 'Healthy', 'Nevi', 'No Finding']
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

import io
import base64
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

@app.route('/api/generate-pdf', methods=['POST'])
def generate_pdf():
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "Пустой запрос"}), 400
            
        disease_info = data.get('disease_data')
        image_data = data.get('image')
        
        if not disease_info or not image_data:
            return jsonify({"success": False, "error": "Данные анализа или фото отсутствуют"}), 400

        # Создаем PDF в оперативной памяти
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        width, height = letter

        # Настройка шрифта (Кириллица)
        font_path = "C:/Windows/Fonts/arial.ttf" # Для Windows
        font_name = "ArialCustom"
        try:
            pdfmetrics.registerFont(TTFont(font_name, font_path))
        except:
            font_name = "Helvetica" # Если шрифт не найден, будут квадратики, но файл создастся

        # --- Отрисовка ---
        # Заголовок
        c.setFont(font_name, 26)
        c.drawCentredString(width / 2, height - 50, "ainala.ai") # Исправлено на Centred
        
        c.setFont(font_name, 14)
        c.drawCentredString(width / 2, height - 75, "Медицинский ИИ-отчет")
        c.line(50, height - 85, width - 50, height - 85)

        # Картинка
        try:
            if "," in image_data:
                image_data = image_data.split(",")[1]
            img_bytes = base64.b64decode(image_data)
            img_reader = ImageReader(io.BytesIO(img_bytes))
            c.drawImage(img_reader, (width - 250) / 2, height - 350, width=250, height=250, preserveAspectRatio=True)
        except Exception as e:
            print(f"Ошибка изображения: {e}")

        # Результаты
        name = disease_info.get('disease') or disease_info.get('name') or "Не определено"
        prob = disease_info.get('confidence') or disease_info.get('probability') or 0
        
        c.setFont(font_name, 18)
        c.drawCentredString(width / 2, height - 380, f"Результат: {name}")
        c.setFont(font_name, 14)
        c.drawCentredString(width / 2, height - 405, f"Вероятность: {prob}%")

        # Подвал
        c.setFont(font_name, 10)
        c.drawCentredString(width / 2, 50, f"Дата анализа: {datetime.now().strftime('%d.%m.%Y %H:%M')}")

        c.showPage()
        c.save()

        # Подготовка буфера для отправки
        pdf_buffer.seek(0)
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"ainala_report_{datetime.now().strftime('%H%M%S')}.pdf"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/diseases', methods=['GET'])
def get_diseases():
    try:
        # Получаем параметры поиска и категории из запроса
        search_query = request.args.get('search', '').lower()
        category = request.args.get('category', 'Все')

        filtered_diseases = DISEASES_DATABASE

        # Фильтрация по категории
        if category != 'Все':
            filtered_diseases = [d for d in filtered_diseases if d['category'] == category]

        # Фильтрация по поисковому слову
        if search_query:
            filtered_diseases = [
                d for d in filtered_diseases 
                if search_query in d['name'].lower() or search_query in d['description'].lower()
            ]

        return jsonify({
            "success": True,
            "diseases": filtered_diseases
        })
    except Exception as e:
        print(f"Error in get_diseases: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader = False, port=5000)