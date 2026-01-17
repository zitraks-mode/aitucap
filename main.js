// Глобальные переменные
let currentImage = null;
let currentAnalysis = null;
let currentPage = 'home';
let chatHistory = [];
let currentAnalysisId = null;

// Показ страниц
function showPage(pageId) {
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    
    const pageElement = document.getElementById(`${pageId}-page`);
    if (pageElement) {
        pageElement.classList.add('active');
    }
    
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
        if (link.textContent.includes(getPageName(pageId))) {
            link.classList.add('active');
        }
    });
    
    currentPage = pageId;
    
    if (pageId === 'history') {
        loadHistory();
    } else if (pageId === 'library') {
        loadDiseases();
    } else if (pageId === 'scanner') {
        resetScanner();
    } else if (pageId === 'chat') {
        resetChat();
    }
}

function getPageName(pageId) {
    const pageNames = {
        'home': 'Главная',
        'scanner': 'Сканер',
        'xray': 'Рентген',
        'library': 'Библиотека',
        'chat': 'ИИ Чат',
        'history': 'История'
    };
    return pageNames[pageId] || '';
}

// Сброс сканера
function resetScanner() {
    const uploadArea = document.getElementById('uploadArea');
    const previewContainer = document.getElementById('previewContainer');
    const resultsContainer = document.getElementById('resultsContainer');
    const hospitalsSection = document.getElementById('hospitalsSection');
    const fileInput = document.getElementById('fileInput');
    
    currentImage = null;
    currentAnalysis = null;
    currentAnalysisId = null;
    
    if (uploadArea) uploadArea.style.display = 'block';
    if (previewContainer) previewContainer.style.display = 'none';
    if (resultsContainer) resultsContainer.style.display = 'none';
    if (hospitalsSection) hospitalsSection.style.display = 'none';
    
    if (fileInput) {
        fileInput.value = '';
    }
}

// Сброс чата
function resetChat() {
    const chatMessages = document.getElementById('chatMessages');
    if (chatMessages) {
        // Добавляем приветственное сообщение при каждом заходе в чат
        if (!chatMessages.querySelector('.ai-message')) {
            addChatMessage('Привет! Я ваш медицинский ассистент. Задайте мне вопрос о симптомах, заболеваниях или здоровье. Помните, что я не заменяю врача, а лишь предоставляю информацию.', 'ai');
        }
    }
}

// Меню на мобильных
function toggleMenu() {
    const nav = document.querySelector('.nav');
    nav.classList.toggle('show');
}

// Загрузка изображения
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (file) {
        if (!file.type.match('image.*')) {
            alert('Пожалуйста, выберите файл изображения (JPG, PNG, GIF)');
            return;
        }
        
        if (file.size > 10 * 1024 * 1024) {
            alert('Файл слишком большой. Максимальный размер: 10MB');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = function(e) {
            showPreview(e.target.result);
        };
        reader.onerror = function() {
            alert('Ошибка при чтении файла');
        };
        reader.readAsDataURL(file);
    }
}

// Показ превью
function showPreview(imageData) {
    currentImage = imageData;
    
    const uploadArea = document.getElementById('uploadArea');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    
    if (uploadArea) uploadArea.style.display = 'none';
    if (previewImage) previewImage.src = imageData;
    if (previewContainer) previewContainer.style.display = 'block';
    
    if (previewContainer) {
        previewContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

// Очистка превью
function clearPreview() {
    currentImage = null;
    
    const uploadArea = document.getElementById('uploadArea');
    const previewContainer = document.getElementById('previewContainer');
    const fileInput = document.getElementById('fileInput');
    
    if (uploadArea) uploadArea.style.display = 'block';
    if (previewContainer) previewContainer.style.display = 'none';
    if (fileInput) {
        fileInput.value = '';
    }
    
    const resultsContainer = document.getElementById('resultsContainer');
    const hospitalsSection = document.getElementById('hospitalsSection');
    if (resultsContainer) resultsContainer.style.display = 'none';
    if (hospitalsSection) hospitalsSection.style.display = 'none';
}

// Камера
function capturePhoto() {
    const modal = document.getElementById('cameraModal');
    if (modal) modal.style.display = 'flex';
    
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ 
            video: { 
                facingMode: 'environment',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            } 
        })
        .then(stream => {
            const video = document.getElementById('cameraVideo');
            if (video) {
                video.srcObject = stream;
                video.play();
            }
        })
        .catch(err => {
            console.error('Camera error:', err);
            alert('Не удалось получить доступ к камере: ' + err.message);
            closeCamera();
        });
    } else {
        alert('Ваш браузер не поддерживает доступ к камере');
        closeCamera();
    }
}

function closeCamera() {
    const modal = document.getElementById('cameraModal');
    const video = document.getElementById('cameraVideo');
    
    if (video && video.srcObject) {
        const tracks = video.srcObject.getTracks();
        tracks.forEach(track => {
            track.stop();
        });
        video.srcObject = null;
    }
    
    if (modal) modal.style.display = 'none';
}

function takePhoto() {
    const video = document.getElementById('cameraVideo');
    const canvas = document.getElementById('cameraCanvas');
    if (!video || !canvas) return;
    
    const context = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    
    closeCamera();
    showPreview(imageData);
}

// Анализ изображения
async function analyzeImage() {
    if (!currentImage) {
        alert('Пожалуйста, загрузите изображение');
        return;
    }
    
    const typeOption = document.querySelector('.type-option.active');
    const analysisType = typeOption ? typeOption.dataset.type : 'skin';
    
    const analyzeButton = document.querySelector('.preview-container .btn');
    
    if (!analyzeButton) return;
    
    const originalText = analyzeButton.innerHTML;
    analyzeButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Анализ...';
    analyzeButton.disabled = true;
    
    try {
        console.log('Отправка запроса на анализ...');
        
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: currentImage,
                type: analysisType
            })
        });
        
        console.log('Получен ответ:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Данные ответа:', data);
        
        if (data.success) {
            currentAnalysisId = data.analysis_id;
            currentAnalysis = data.result;
            showResults(data.result);
        } else {
            throw new Error(data.error || 'Неизвестная ошибка анализа');
        }
    } catch (error) {
        console.error('Ошибка анализа:', error);
        
        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            alert('Сервер недоступен. Показываю демо-результаты...');
            showDemoResults(analysisType);
        } else {
            alert('Ошибка при анализе изображения: ' + error.message);
        }
    } finally {
        analyzeButton.innerHTML = originalText;
        analyzeButton.disabled = false;
    }
}

// Показать результаты анализа (обычные изображения)
function showResults(result) {
    const previewContainer = document.getElementById('previewContainer');
    const resultsContainer = document.getElementById('resultsContainer');
    const analysisDate = document.getElementById('analysisDate');
    const analysisType = document.getElementById('analysisType');
    const diseasesList = document.getElementById('diseasesList');
    const recommendationsText = document.getElementById('recommendationsText');
    
    if (previewContainer) previewContainer.style.display = 'none';
    if (resultsContainer) resultsContainer.style.display = 'block';
    
    if (analysisDate) {
        const date = new Date(result.timestamp);
        analysisDate.textContent = `Дата: ${date.toLocaleDateString('ru-RU')} ${date.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' })}`;
    }
    
    if (analysisType) {
        analysisType.textContent = `Тип анализа: ${result.type.charAt(0).toUpperCase() + result.type.slice(1)}`;
    }
    
    if (diseasesList) {
        diseasesList.innerHTML = '';
        
        const disease = result.disease;
        const isNormal = disease.is_normal === true;
        
        // КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: проверяем is_normal
        if (isNormal) {
            // Показываем результат "Норма"
            const diseaseItem = document.createElement('div');
            diseaseItem.className = 'disease-item normal-result';
            
            diseaseItem.innerHTML = `
                <div class="disease-header">
                    <div class="disease-title">
                        <i class="fas fa-check-circle" style="color: #10b981;"></i>
                        <div class="disease-name">Заболеваний не обнаружено</div>
                    </div>
                    <div class="probability-container">
                        <div class="probability good">
                            <span class="probability-value">${disease.probability}%</span>
                        </div>
                        <div class="severity-indicator good">Норма</div>
                    </div>
                </div>
                
                <div class="disease-details">
                    <div class="detail-section">
                        <h4><i class="fas fa-info-circle"></i> Результат</h4>
                        <p>По результатам анализа патологий не выявлено. Продолжайте заботиться о своем здоровье!</p>
                    </div>
                    
                    <div class="detail-section recommendations-section">
                        <h4><i class="fas fa-clipboard-check"></i> Профилактические рекомендации</h4>
                        <ul>
                            <li>Продолжайте вести здоровый образ жизни</li>
                            <li>Регулярно проходите профилактические осмотры</li>
                            <li>Сбалансированно питайтесь</li>
                            <li>Занимайтесь физической активностью</li>
                        </ul>
                    </div>
                </div>
            `;
            
            diseasesList.appendChild(diseaseItem);
            
            if (recommendationsText) {
                recommendationsText.innerHTML = `
                    <div class="success-message">
                        <i class="fas fa-check-circle"></i>
                        <strong>Отличные новости!</strong> Анализ не выявил патологий. Продолжайте следить за своим здоровьем.
                    </div>
                `;
            }
        } else {
            // Показываем заболевание
            let probabilityClass = 'medium';
            let severity = '';
            
            if (disease.probability >= 70) {
                probabilityClass = 'high';
                severity = 'high';
            } else if (disease.probability >= 40) {
                probabilityClass = 'medium';
                severity = 'medium';
            } else {
                probabilityClass = 'low';
                severity = 'low';
            }
            
            let diseaseIcon = 'fa-heartbeat';
            let diseaseColor = '#4f46e5';
            
            if (disease.name.includes('Меланома') || disease.name.includes('Рак')) {
                diseaseIcon = 'fa-exclamation-triangle';
                diseaseColor = '#ef4444';
            } else if (disease.name.includes('Пневмония') || disease.name.includes('Туберкулез')) {
                diseaseIcon = 'fa-lungs-virus';
                diseaseColor = '#f59e0b';
            }
            
            const specialists = disease.specialists || ['Терапевт'];
            
            const diseaseItem = document.createElement('div');
            diseaseItem.className = 'disease-item';
            
            diseaseItem.innerHTML = `
                <div class="disease-header">
                    <div class="disease-title">
                        <i class="fas ${diseaseIcon}" style="color: ${diseaseColor};"></i>
                        <div class="disease-name">${disease.name}</div>
                    </div>
                    <div class="probability-container">
                        <div class="probability ${probabilityClass}">
                            <span class="probability-value">${disease.probability}%</span>
                        </div>
                        <div class="severity-indicator ${severity}">
                            ${severity === 'high' ? 'Высокий риск' : 
                              severity === 'medium' ? 'Средний риск' : 
                              'Низкий риск'}
                        </div>
                    </div>
                </div>
                
                <div class="disease-details">
                    <div class="detail-section">
                        <h4><i class="fas fa-info-circle"></i> Описание</h4>
                        <p>${disease.description || 'Требуется консультация специалиста'}</p>
                    </div>
                    
                    ${disease.symptoms ? `
                    <div class="detail-section">
                        <h4><i class="fas fa-exclamation-circle"></i> Симптомы</h4>
                        <ul>
                            ${disease.symptoms.map(s => `<li>${s}</li>`).join('')}
                        </ul>
                    </div>
                    ` : ''}
                    
                    ${disease.causes ? `
                    <div class="detail-section">
                        <h4><i class="fas fa-search"></i> Причины</h4>
                        <ul>
                            ${disease.causes.map(c => `<li>${c}</li>`).join('')}
                        </ul>
                    </div>
                    ` : ''}
                    
                    ${disease.danger ? `
                    <div class="detail-section">
                        <h4><i class="fas fa-shield-alt"></i> Опасность</h4>
                        <div class="danger-level ${probabilityClass}">
                            <i class="fas ${probabilityClass === 'high' ? 'fa-exclamation-triangle' : 
                                           probabilityClass === 'medium' ? 'fa-exclamation-circle' : 
                                           'fa-info-circle'}"></i>
                            <span>${disease.danger}</span>
                        </div>
                    </div>
                    ` : ''}
                    
                    <div class="detail-section specialists-section">
                        <h4><i class="fas fa-user-md"></i> К кому обратиться</h4>
                        <div class="specialists-list">
                            ${specialists.map(spec => `
                                <span class="specialist-tag">
                                    <i class="fas fa-stethoscope"></i> ${spec}
                                </span>
                            `).join('')}
                        </div>
                    </div>
                    
                    ${disease.recommendations ? `
                    <div class="detail-section recommendations-section">
                        <h4><i class="fas fa-clipboard-check"></i> Рекомендации</h4>
                        <ul>
                            ${disease.recommendations.map(r => `<li>${r}</li>`).join('')}
                        </ul>
                    </div>
                    ` : ''}
                    
                    ${disease.hospitals && disease.hospitals.length > 0 ? `
                    <div class="detail-section">
                        <h4><i class="fas fa-hospital"></i> Рекомендованные больницы</h4>
                        <div class="hospitals-mini-list">
                            ${disease.hospitals.slice(0, 3).map(h => `
                                <div class="hospital-mini-item">
                                    <strong>${h.name}</strong>
                                    <div>${h.city}, ${h.address}</div>
                                    <div>${h.phone}</div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                    ` : ''}
                </div>
                
                <div class="disease-footer">
                    <div class="confidence-level">
                        <span>Доверие анализу:</span>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${disease.probability}%"></div>
                        </div>
                        <span class="confidence-percent">${disease.probability}%</span>
                    </div>
                </div>
            `;
            
            diseasesList.appendChild(diseaseItem);
            
            if (recommendationsText) {
                const hasSeriousDisease = disease.probability >= 70;
                
                if (hasSeriousDisease) {
                    recommendationsText.innerHTML = `
                        <div class="urgent-warning">
                            <i class="fas fa-exclamation-triangle"></i>
                            <strong>ВАЖНО:</strong> Обнаружены признаки серьезного заболевания. Рекомендуется СРОЧНО обратиться к врачу!
                        </div>
                    `;
                } else {
                    recommendationsText.textContent = 'Рекомендуется обратиться к врачу для подтверждения диагноза и получения профессионального лечения.';
                }
            }
        }
    }
    
    if (resultsContainer) {
        resultsContainer.scrollIntoView({ behavior: 'smooth' });
    }
}

// Демо-результаты для обычного анализа
function showDemoResults(analysisType) {
    const demoData = {
        disease: {
            name: analysisType === 'skin' ? 'Акне' : 
                  analysisType === 'eyes' ? 'Конъюнктивит' : 
                  analysisType === 'nails' ? 'Грибок ногтей' : 'Пневмония',
            probability: Math.floor(Math.random() * 30) + 60,
            is_normal: false,
            description: 'Демонстрационный результат для тестирования интерфейса',
            symptoms: ['Симптом 1', 'Симптом 2', 'Симптом 3'],
            causes: ['Причина 1', 'Причина 2'],
            danger: 'Низкая/средняя опасность',
            specialists: ['Дерматолог', 'Терапевт'],
            recommendations: ['Обратиться к врачу', 'Пройти обследование'],
            hospitals: []
        },
        timestamp: new Date().toISOString(),
        type: analysisType
    };
    
    currentAnalysisId = Math.floor(Math.random() * 1000) + 1;
    currentAnalysis = demoData;
    showResults(demoData);
}

// Анализ рентгена
async function analyzeXray(imageData) {
    const xrayResults = document.getElementById('xrayResults');
    const xrayPreview = document.getElementById('xrayPreview');
    
    if (xrayResults) xrayResults.style.display = 'none';
    if (xrayPreview) xrayPreview.style.display = 'none';
    
    const loadingIndicator = document.getElementById('xrayLoading');
    if (loadingIndicator) {
        loadingIndicator.style.display = 'block';
        loadingIndicator.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Анализ рентгена...';
    }
    
    try {
        console.log('Отправка рентгена на анализ...');
        
        const response = await fetch('/api/analyze-xray', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageData
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        
        const data = await response.json();
        
        if (loadingIndicator) loadingIndicator.style.display = 'none';
        
        if (data.success) {
            currentAnalysisId = data.analysis_id;
            currentAnalysis = data.result;
            showXrayResults(data.result);
        } else {
            throw new Error(data.error || 'Неизвестная ошибка анализа');
        }
    } catch (error) {
        console.error('Ошибка анализа рентгена:', error);
        
        if (loadingIndicator) loadingIndicator.style.display = 'none';
        
        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            alert('Сервер недоступен. Показываю демо-результаты рентгена...');
            showDemoXrayResults();
        } else {
            alert('Ошибка при анализе рентгена: ' + error.message);
        }
    }
}

// Обработка загрузки рентгена
function handleXrayUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    if (!file.type.match('image.*')) {
        alert('Пожалуйста, выберите файл изображения (JPG, PNG)');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = function(e) {
        // Показываем превью
        const xrayPreview = document.getElementById('xrayPreview');
        const xrayPreviewImg = document.getElementById('xrayPreviewImg');
        
        if (xrayPreviewImg) xrayPreviewImg.src = e.target.result;
        if (xrayPreview) xrayPreview.style.display = 'block';
        
        // Анализируем изображение
        analyzeXray(e.target.result);
    };
    reader.readAsDataURL(file);
}

// Показать результаты анализа рентгена
function showXrayResults(result) {
    const xrayResults = document.getElementById('xrayResults');
    if (!xrayResults) return;
    
    xrayResults.style.display = 'block';
    
    const isNormal = result.disease?.is_normal === true;
    
    if (isNormal) {
        xrayResults.innerHTML = `
            <div class="results-header">
                <h2><i class="fas fa-x-ray"></i> Результаты рентген-анализа</h2>
                <div class="analysis-info">
                    <span class="date">${new Date().toLocaleDateString('ru-RU')}</span>
                    <span class="type">Рентген-анализ</span>
                </div>
            </div>
            
            <div class="diseases-list">
                <div class="disease-item normal-result">
                    <div class="disease-header">
                        <div class="disease-title">
                            <i class="fas fa-check-circle" style="color: #10b981;"></i>
                            <div class="disease-name">Патологий не обнаружено</div>
                        </div>
                        <div class="probability-container">
                            <div class="probability good">
                                <span class="probability-value">${result.disease.probability || 95}%</span>
                            </div>
                            <div class="severity-indicator good">Норма</div>
                        </div>
                    </div>
                    
                    <div class="disease-details">
                        <div class="detail-section">
                            <h4><i class="fas fa-info-circle"></i> Результат</h4>
                            <p>Рентген-снимок не выявил патологий легких или костных структур.</p>
                        </div>
                        
                        <div class="detail-section recommendations-section">
                            <h4><i class="fas fa-clipboard-check"></i> Рекомендации</h4>
                            <ul>
                                <li>Продолжайте регулярные профилактические осмотры</li>
                                <li>Избегайте курения и вредных производственных факторов</li>
                                <li>При появлении кашля, одышки или боли в груди обратитесь к врачу</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="actions-section">
                <button class="btn btn-primary" onclick="generateXrayPDF()">
                    <i class="fas fa-file-pdf"></i> Скачать PDF отчет
                </button>
            </div>
        `;
    } else {
        const disease = result.disease;
        const probability = disease.probability || 75;
        
        let probabilityClass = 'medium';
        let severity = 'medium';
        
        if (probability >= 70) {
            probabilityClass = 'high';
            severity = 'high';
        } else if (probability >= 40) {
            probabilityClass = 'medium';
            severity = 'medium';
        } else {
            probabilityClass = 'low';
            severity = 'low';
        }
        
        xrayResults.innerHTML = `
            <div class="results-header">
                <h2><i class="fas fa-x-ray"></i> Результаты рентген-анализа</h2>
                <div class="analysis-info">
                    <span class="date">${new Date().toLocaleDateString('ru-RU')}</span>
                    <span class="type">Рентген-анализ</span>
                </div>
            </div>
            
            <div class="diseases-list">
                <div class="disease-item">
                    <div class="disease-header">
                        <div class="disease-title">
                            <i class="fas fa-lungs-virus" style="color: #f59e0b;"></i>
                            <div class="disease-name">${disease.name || 'Обнаружена патология'}</div>
                        </div>
                        <div class="probability-container">
                            <div class="probability ${probabilityClass}">
                                <span class="probability-value">${probability}%</span>
                            </div>
                            <div class="severity-indicator ${severity}">
                                ${severity === 'high' ? 'Высокий риск' : 
                                  severity === 'medium' ? 'Средний риск' : 
                                  'Низкий риск'}
                            </div>
                        </div>
                    </div>
                    
                    <div class="disease-details">
                        <div class="detail-section">
                            <h4><i class="fas fa-info-circle"></i> Описание</h4>
                            <p>${disease.description || 'Обнаружены признаки патологии на рентгеновском снимке. Требуется консультация специалиста.'}</p>
                        </div>
                        
                        ${disease.symptoms ? `
                        <div class="detail-section">
                            <h4><i class="fas fa-exclamation-circle"></i> Симптомы</h4>
                            <ul>
                                ${disease.symptoms.map(s => `<li>${s}</li>`).join('')}
                            </ul>
                        </div>
                        ` : ''}
                        
                        ${disease.location ? `
                        <div class="detail-section">
                            <h4><i class="fas fa-location-dot"></i> Локализация</h4>
                            <p>${disease.location}</p>
                        </div>
                        ` : ''}
                        
                        <div class="detail-section specialists-section">
                            <h4><i class="fas fa-user-md"></i> К кому обратиться</h4>
                            <div class="specialists-list">
                                <span class="specialist-tag">
                                    <i class="fas fa-stethoscope"></i> Рентгенолог
                                </span>
                                <span class="specialist-tag">
                                    <i class="fas fa-stethoscope"></i> Пульмонолог
                                </span>
                                <span class="specialist-tag">
                                    <i class="fas fa-stethoscope"></i> Терапевт
                                </span>
                            </div>
                        </div>
                        
                        ${disease.recommendations ? `
                        <div class="detail-section recommendations-section">
                            <h4><i class="fas fa-clipboard-check"></i> Рекомендации</h4>
                            <ul>
                                ${disease.recommendations.map(r => `<li>${r}</li>`).join('')}
                            </ul>
                        </div>
                        ` : ''}
                    </div>
                </div>
            </div>
            
            <div class="actions-section">
                <button class="btn btn-primary" onclick="generateXrayPDF()">
                    <i class="fas fa-file-pdf"></i> Скачать PDF отчет
                </button>
                <button class="btn btn-secondary" onclick="findHospitalsForDisease('${disease.name || 'Рентген-патология'}')">
                    <i class="fas fa-hospital"></i> Найти больницы
                </button>
            </div>
        `;
    }
    
    xrayResults.scrollIntoView({ behavior: 'smooth' });
}

// Демо-результаты для рентгена
function showDemoXrayResults() {
    const demoData = {
        disease: {
            name: 'Пневмония',
            probability: 85,
            is_normal: false,
            description: 'Обнаружены инфильтративные изменения в нижней доле правого легкого, характерные для воспалительного процесса.',
            symptoms: ['Кашель с мокротой', 'Повышение температуры', 'Одышка', 'Боль в груди при дыхании'],
            location: 'Нижняя доля правого легкого',
            specialists: ['Пульмонолог', 'Рентгенолог', 'Терапевт'],
            recommendations: [
                'Срочно обратиться к врачу',
                'Пройти КТ грудной клетки для уточнения',
                'Сдать общий анализ крови',
                'Начать антибактериальную терапию по назначению врача'
            ]
        },
        timestamp: new Date().toISOString(),
        type: 'xray'
    };
    
    currentAnalysisId = Math.floor(Math.random() * 1000) + 1;
    currentAnalysis = demoData;
    showXrayResults(demoData);
}

// Генерация PDF для обычного анализа
async function generatePDF() {
    if (!currentAnalysisId || !currentAnalysis) {
        alert('Сначала выполните анализ');
        return;
    }
    
    const generateBtn = document.querySelector('.actions-section .btn-primary');
    if (!generateBtn) return;
    
    const originalText = generateBtn.innerHTML;
    generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Генерация PDF...';
    generateBtn.disabled = true;
    
    try {
        // Создаем HTML для PDF
        const disease = currentAnalysis.disease;
        const isNormal = disease.is_normal === true;
        
        let pdfContent = `
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Медицинский отчет</title>
                <style>
                    body { font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }
                    .header { text-align: center; margin-bottom: 30px; border-bottom: 2px solid #4f46e5; padding-bottom: 20px; }
                    .header h1 { color: #4f46e5; margin: 0; }
                    .info { margin: 20px 0; padding: 15px; background: #f3f4f6; border-radius: 5px; }
                    .result { margin: 30px 0; padding: 20px; border-left: 4px solid #4f46e5; background: #f8fafc; }
                    .normal { border-left-color: #10b981; }
                    .abnormal { border-left-color: #ef4444; }
                    .probability { font-weight: bold; font-size: 1.2em; margin: 10px 0; }
                    .recommendations { margin-top: 30px; padding: 15px; background: #f0f9ff; border-radius: 5px; }
                    .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 0.9em; color: #666; text-align: center; }
                    ul { padding-left: 20px; }
                    li { margin: 5px 0; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Медицинский отчет о диагностике</h1>
                    <p>Отчет сгенерирован: ${new Date().toLocaleDateString('ru-RU')} ${new Date().toLocaleTimeString('ru-RU')}</p>
                    <p>ID анализа: ${currentAnalysisId}</p>
                </div>
                
                <div class="info">
                    <h3>Информация об анализе</h3>
                    <p><strong>Тип анализа:</strong> ${currentAnalysis.type ? currentAnalysis.type.charAt(0).toUpperCase() + currentAnalysis.type.slice(1) : 'Неизвестно'}</p>
                    <p><strong>Дата анализа:</strong> ${new Date(currentAnalysis.timestamp || Date.now()).toLocaleDateString('ru-RU')}</p>
                </div>
        `;
        
        if (isNormal) {
            pdfContent += `
                <div class="result normal">
                    <h2 style="color: #10b981;">✓ Заболеваний не обнаружено</h2>
                    <p>По результатам анализа патологий не выявлено.</p>
                    <div class="probability">Вероятность нормы: ${disease.probability || 95}%</div>
                </div>
                
                <div class="recommendations">
                    <h3>Профилактические рекомендации:</h3>
                    <ul>
                        <li>Продолжайте вести здоровый образ жизни</li>
                        <li>Регулярно проходите профилактические осмотры</li>
                        <li>Сбалансированно питайтесь</li>
                        <li>Занимайтесь физической активностью</li>
                    </ul>
                </div>
            `;
        } else {
            pdfContent += `
                <div class="result abnormal">
                    <h2 style="color: #ef4444;">⚠️ Обнаружено: ${disease.name}</h2>
                    <p>${disease.description || 'Обнаружены признаки заболевания. Требуется консультация специалиста.'}</p>
                    <div class="probability">Вероятность: ${disease.probability || 75}%</div>
                    
                    ${disease.symptoms ? `
                    <h3>Симптомы:</h3>
                    <ul>
                        ${disease.symptoms.map(s => `<li>${s}</li>`).join('')}
                    </ul>
                    ` : ''}
                    
                    ${disease.causes ? `
                    <h3>Возможные причины:</h3>
                    <ul>
                        ${disease.causes.map(c => `<li>${c}</li>`).join('')}
                    </ul>
                    ` : ''}
                </div>
                
                <div class="info">
                    <h3>Рекомендуемые специалисты:</h3>
                    <p>${(disease.specialists || ['Терапевт']).join(', ')}</p>
                </div>
                
                ${disease.recommendations ? `
                <div class="recommendations">
                    <h3>Рекомендации:</h3>
                    <ul>
                        ${disease.recommendations.map(r => `<li>${r}</li>`).join('')}
                    </ul>
                </div>
                ` : ''}
            `;
        }
        
        pdfContent += `
                <div class="footer">
                    <p><strong>ВАЖНО:</strong> Данный отчет сгенерирован системой ИИ-диагностики и не заменяет консультацию врача.</p>
                    <p>Для точной диагностики и лечения обратитесь к квалифицированному медицинскому специалисту.</p>
                    <p>Отчет сгенерирован системой Medical AI Assistant</p>
                </div>
            </body>
            </html>
        `;
        
        // Отправляем HTML на сервер для генерации PDF
        const response = await fetch('/api/generate-pdf', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                analysis_id: currentAnalysisId,
                html_content: pdfContent,
                type: 'regular'
            })
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `medical_report_${currentAnalysisId}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } else {
            const error = await response.json();
            throw new Error(error.error || 'Ошибка генерации PDF');
        }
    } catch (error) {
        console.error('PDF generation error:', error);
        alert('Ошибка при генерации PDF: ' + error.message);
    } finally {
        generateBtn.innerHTML = originalText;
        generateBtn.disabled = false;
    }
}

// Генерация PDF для рентгена
async function generateXrayPDF() {
    if (!currentAnalysisId || !currentAnalysis) {
        alert('Сначала выполните анализ рентгена');
        return;
    }
    
    const generateBtn = document.querySelector('#xrayResults .btn-primary');
    if (!generateBtn) return;
    
    const originalText = generateBtn.innerHTML;
    generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Генерация PDF...';
    generateBtn.disabled = true;
    
    try {
        // Создаем HTML для PDF
        const disease = currentAnalysis.disease;
        const isNormal = disease.is_normal === true;
        
        let pdfContent = `
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Рентгенологический отчет</title>
                <style>
                    body { font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }
                    .header { text-align: center; margin-bottom: 30px; border-bottom: 2px solid #4f46e5; padding-bottom: 20px; }
                    .header h1 { color: #4f46e5; margin: 0; }
                    .info { margin: 20px 0; padding: 15px; background: #f3f4f6; border-radius: 5px; }
                    .result { margin: 30px 0; padding: 20px; border-left: 4px solid #4f46e5; background: #f8fafc; }
                    .normal { border-left-color: #10b981; }
                    .abnormal { border-left-color: #ef4444; }
                    .probability { font-weight: bold; font-size: 1.2em; margin: 10px 0; }
                    .recommendations { margin-top: 30px; padding: 15px; background: #f0f9ff; border-radius: 5px; }
                    .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 0.9em; color: #666; text-align: center; }
                    ul { padding-left: 20px; }
                    li { margin: 5px 0; }
                    .xray-note { background: #fff3cd; padding: 10px; border-radius: 5px; border-left: 4px solid #ffc107; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Рентгенологический отчет</h1>
                    <p>Отчет сгенерирован: ${new Date().toLocaleDateString('ru-RU')} ${new Date().toLocaleTimeString('ru-RU')}</p>
                    <p>ID анализа: ${currentAnalysisId}</p>
                </div>
                
                <div class="info">
                    <h3>Информация об исследовании</h3>
                    <p><strong>Тип исследования:</strong> Рентгенография</p>
                    <p><strong>Дата исследования:</strong> ${new Date(currentAnalysis.timestamp || Date.now()).toLocaleDateString('ru-RU')}</p>
                </div>
                
                <div class="xray-note">
                    <p><strong>Примечание:</strong> Данный отчет создан системой ИИ-анализа рентгеновских снимков. Для окончательного диагноза требуется заключение врача-рентгенолога.</p>
                </div>
        `;
        
        if (isNormal) {
            pdfContent += `
                <div class="result normal">
                    <h2 style="color: #10b981;">✓ Патологий не обнаружено</h2>
                    <p>Рентген-снимок не выявил патологических изменений.</p>
                    <div class="probability">Вероятность нормы: ${disease.probability || 95}%</div>
                    
                    <h3>Описание:</h3>
                    <p>Легочные поля прозрачные, легочный рисунок не изменен. Корни структурны. Диафрагма расположена обычно. Сердечная тень не расширена.</p>
                </div>
            `;
        } else {
            pdfContent += `
                <div class="result abnormal">
                    <h2 style="color: #ef4444;">⚠️ Обнаружено: ${disease.name}</h2>
                    <p>${disease.description || 'Обнаружены патологические изменения на рентгеновском снимке.'}</p>
                    <div class="probability">Вероятность: ${disease.probability || 75}%</div>
                    
                    ${disease.location ? `
                    <h3>Локализация:</h3>
                    <p>${disease.location}</p>
                    ` : ''}
                    
                    ${disease.symptoms ? `
                    <h3>Клинические проявления:</h3>
                    <ul>
                        ${disease.symptoms.map(s => `<li>${s}</li>`).join('')}
                    </ul>
                    ` : ''}
                </div>
            `;
        }
        
        pdfContent += `
                <div class="info">
                    <h3>Рекомендуемые действия:</h3>
                    <p><strong>Специалисты для консультации:</strong> Рентгенолог, Пульмонолог, Терапевт</p>
                </div>
                
                ${disease.recommendations ? `
                <div class="recommendations">
                    <h3>Дальнейшие рекомендации:</h3>
                    <ul>
                        ${disease.recommendations.map(r => `<li>${r}</li>`).join('')}
                    </ul>
                </div>
                ` : `
                <div class="recommendations">
                    <h3>Стандартные рекомендации:</h3>
                    <ul>
                        <li>Обратиться к врачу-рентгенологу для получения заключения</li>
                        <li>Проконсультироваться с терапевтом или пульмонологом</li>
                        <li>При необходимости выполнить КТ грудной клетки</li>
                        <li>Сдать клинические анализы крови</li>
                    </ul>
                </div>
                `}
                
                <div class="footer">
                    <p><strong>ВАЖНО:</strong> Данный отчет является предварительным. Окончательное заключение дает врач-рентгенолог на основании изучения оригинала снимка.</p>
                    <p>Отчет сгенерирован системой Medical AI Assistant - Xray Module</p>
                </div>
            </body>
            </html>
        `;
        
        // Отправляем HTML на сервер для генерации PDF
        const response = await fetch('/api/generate-pdf', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                analysis_id: currentAnalysisId,
                html_content: pdfContent,
                type: 'xray'
            })
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `xray_report_${currentAnalysisId}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } else {
            const error = await response.json();
            throw new Error(error.error || 'Ошибка генерации PDF');
        }
    } catch (error) {
        console.error('Xray PDF generation error:', error);
        alert('Ошибка при генерации PDF: ' + error.message);
    } finally {
        generateBtn.innerHTML = originalText;
        generateBtn.disabled = false;
    }
}

// Чат с ИИ (исправленная версия)
async function sendChatMessage() {
    const input = document.getElementById('chatInput');
    if (!input) return;
    
    const message = input.value.trim();
    
    if (!message) return;
    
    addChatMessage(message, 'user');
    input.value = '';
    
    const loadingId = addLoadingMessage();
    
    try {
        // Проверяем, есть ли доступ к реальному API чата
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                history: chatHistory.slice(-5) // Отправляем последние 5 сообщений для контекста
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        
        const data = await response.json();
        
        removeLoadingMessage(loadingId);
        
        if (data.success && data.message) {
            const aiMessage = data.message;
            addChatMessage(aiMessage, 'ai');
            
            // Обновляем историю чата
            chatHistory.push({
                text: message,
                sender: 'user',
                timestamp: new Date().toISOString()
            });
            
            chatHistory.push({
                text: aiMessage,
                sender: 'ai',
                timestamp: new Date().toISOString()
            });
            
            // Ограничиваем историю 20 сообщениями
            if (chatHistory.length > 20) {
                chatHistory = chatHistory.slice(-20);
            }
            
            // Сохраняем историю в localStorage
            localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
            
        } else {
            throw new Error('Некорректный ответ от сервера');
        }
        
    } catch (error) {
        console.error('Chat error:', error);
        removeLoadingMessage(loadingId);
        
        // Только если ошибка сети, используем демо-ответы
        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            console.log('Использую демо-режим чата');
            
            // Интеллектуальные демо-ответы в зависимости от вопроса
            let demoResponse = getIntelligentDemoResponse(message);
            addChatMessage(demoResponse, 'ai');
            
            chatHistory.push({
                text: message,
                sender: 'user',
                timestamp: new Date().toISOString()
            });
            
            chatHistory.push({
                text: demoResponse,
                sender: 'ai',
                timestamp: new Date().toISOString()
            });
        } else {
            // Для других ошибок показываем сообщение об ошибке
            addChatMessage('Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже.', 'ai');
        }
    }
}

// Интеллектуальные демо-ответы для чата
function getIntelligentDemoResponse(question) {
    const lowerQuestion = question.toLowerCase();
    
    // Распознавание тем вопросов
    if (lowerQuestion.includes('симптом') || lowerQuestion.includes('болит') || lowerQuestion.includes('температура')) {
        return `Основываясь на вашем описании "${question}", я рекомендую:
        
1. **Немедленно обратитесь к врачу** для точной диагностики
2. **Измерьте температуру тела** - это важный показатель
3. **Обратите внимание на сопутствующие симптомы**: слабость, головная боль, тошнота
4. **До визита к врачу**: пейте больше жидкости, отдыхайте

⚠️ Помните: я не могу ставить диагнозы. Моя задача - предоставить общую информацию.`;
    
    } else if (lowerQuestion.includes('аллерги') || lowerQuestion.includes('сыпь') || lowerQuestion.includes('зуд')) {
        return `Вопрос об аллергических реакциях: "${question}"

**Рекомендации:**
• Исключить контакт с возможными аллергенами
• Проконсультироваться с аллергологом или дерматологом
• Ведите дневник питания и симптомов
• При сильной реакции (отек, затрудненное дыхание) - срочно вызовите скорую

Для точной диагностики необходим очный осмотр специалиста.`;
    
    } else if (lowerQuestion.includes('простуда') || lowerQuestion.includes('грипп') || lowerQuestion.includes('орви')) {
        return `Вопрос о простудных заболеваниях: "${question}"

**Общие рекомендации при ОРВИ:**
1. Постельный режим
2. Обильное теплое питье
3. Проветривание помещения
4. Измерение температуры каждые 4-6 часов

**Когда обратиться к врачу:**
• Температура выше 38.5°C более 3 дней
• Сильная головная боль
• Затрудненное дыхание
• Симптомы не улучшаются через 5-7 дней

Запишитесь на прием к терапевту.`;
    
    } else if (lowerQuestion.includes('анализ') || lowerQuestion.includes('результат') || lowerQuestion.includes('обследовани')) {
        return `Вопрос об анализах и обследованиях: "${question}"

**Важно:**
• Результаты анализов должен интерпретировать врач
• Разные лаборатории могут иметь разные нормы
• На результаты влияют многие факторы (питание, стресс, время суток)

**Что делать:**
1. Записаться на прием к врачу со всеми результатами
2. Не пытаться ставить диагноз самостоятельно
3. Обсудить с врачом необходимость дополнительных исследований

Для конкретной консультации обратитесь к терапевту или профильному специалисту.`;
    
    } else if (lowerQuestion.includes('прививк') || lowerQuestion.includes('вакцин') || lowerQuestion.includes('иммунизац')) {
        return `Вопрос о вакцинации: "${question}"

**Общая информация о вакцинации:**
• Вакцинация - эффективный метод профилактики инфекционных заболеваний
• График прививок составляется индивидуально
• Противопоказания определяет врач

**Рекомендации:**
1. Проконсультироваться с терапевтом или иммунологом
2. Сообщить врачу об аллергиях и хронических заболеваниях
3. Следовать национальному календарю прививок

Для составления индивидуального плана вакцинации обратитесь в поликлинику.`;
    
    } else if (lowerQuestion.includes('диет') || lowerQuestion.includes('питани') || lowerQuestion.includes('еда')) {
        return `Вопрос о питании: "${question}"

**Общие принципы здорового питания:**
• Сбалансированный рацион с овощами и фруктами
• Достаточное количество белка
• Ограничение быстрых углеводов и трансжиров
• Режим питания (3-5 раз в день)

**Для индивидуальной диеты:**
• Проконсультируйтесь с диетологом
• Сдайте анализы на пищевые аллергии
• Учитывайте сопутствующие заболевания

Помните: диета должна быть назначена врачом после обследования.`;
    
    } else {
        // Общий ответ для других вопросов
        return `Спасибо за ваш вопрос: "${question}"

Как медицинский ассистент, я могу предоставить общую информацию, но не могу:
• Ставить диагнозы
• Назначать лечение
• Заменять очную консультацию врача

**Рекомендую:**
1. Записаться на прием к терапевту
2. Подробно описать все симптомы
3. Пройти необходимые обследования
4. Следовать рекомендациям врача

Для получения конкретной медицинской помощи обратитесь в медицинское учреждение.`;
    }
}

function addChatMessage(text, sender) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const timestamp = new Date().toLocaleTimeString('ru-RU', { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-${sender === 'user' ? 'user' : 'robot'}"></i>
        </div>
        <div class="message-content">
            <div class="message-header">
                <span class="sender">${sender === 'user' ? 'Вы' : 'ИИ-ассистент'}</span>
                <span class="timestamp">${timestamp}</span>
            </div>
            <div class="message-text">
                ${formatMessageText(text)}
            </div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addLoadingMessage() {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return null;
    
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message ai-message';
    loadingDiv.id = 'loading-message';
    
    loadingDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="message-content">
            <div class="message-header">
                <span class="sender">ИИ-ассистент</span>
            </div>
            <div class="message-text">
                <p><i class="fas fa-spinner fa-spin"></i> Думаю...</p>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return 'loading-message';
}

function removeLoadingMessage(id) {
    if (!id) return;
    const loadingMessage = document.getElementById(id);
    if (loadingMessage) {
        loadingMessage.remove();
    }
}

function formatMessageText(text) {
    return text
        .split('\n')
        .map(paragraph => {
            if (paragraph.trim() === '') return '';
            
            if (paragraph.trim().startsWith('- ') || paragraph.trim().startsWith('* ') || paragraph.trim().match(/^\d+\./)) {
                return `<p style="margin: 0.5em 0;">${paragraph}</p>`;
            }
            
            if (paragraph.includes(':')) {
                const parts = paragraph.split(':');
                if (parts.length > 1 && parts[0].length < 50) {
                    return `<p><strong>${parts[0]}:</strong> ${parts.slice(1).join(':')}</p>`;
                }
            }
            
            return `<p>${paragraph}</p>`;
        })
        .join('');
}

function handleChatKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendChatMessage();
    }
}

function askQuickQuestion(question) {
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.value = question;
        sendChatMessage();
    }
}

// Загрузка истории чата из localStorage при запуске
function loadChatHistory() {
    try {
        const savedHistory = localStorage.getItem('chatHistory');
        if (savedHistory) {
            chatHistory = JSON.parse(savedHistory);
            
            // Восстанавливаем последние 10 сообщений
            const chatMessages = document.getElementById('chatMessages');
            if (chatMessages) {
                chatMessages.innerHTML = '';
                
                const lastMessages = chatHistory.slice(-10);
                lastMessages.forEach(msg => {
                    addChatMessage(msg.text, msg.sender);
                });
            }
        }
    } catch (error) {
        console.error('Error loading chat history:', error);
    }
}

// Очистка истории чата
function clearChatHistory() {
    if (confirm('Вы уверены, что хотите очистить историю чата?')) {
        chatHistory = [];
        localStorage.removeItem('chatHistory');
        
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            chatMessages.innerHTML = '';
            // Добавляем приветственное сообщение
            addChatMessage('Привет! Я ваш медицинский ассистент. Задайте мне вопрос о симптомах, заболеваниях или здоровье. Помните, что я не заменяю врача, а лишь предоставляю информацию.', 'ai');
        }
    }
}

// Библиотека заболеваний
async function loadDiseases() {
    const diseasesGrid = document.getElementById('diseasesGrid');
    if (!diseasesGrid) return;
    
    diseasesGrid.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i> Загрузка заболеваний...</div>';
    
    try {
        const response = await fetch('/api/diseases');
        if (!response.ok) throw new Error('Network error');
        
        const data = await response.json();
        displayDiseases(data.diseases || []);
    } catch (error) {
        console.error('Error loading diseases:', error);
        displayDiseases([]);
    }
}

function displayDiseases(diseases) {
    const diseasesGrid = document.getElementById('diseasesGrid');
    if (!diseasesGrid) return;
    
    if (!diseases || diseases.length === 0) {
        diseasesGrid.innerHTML = '<p class="empty">Заболевания не найдены</p>';
        return;
    }
    
    diseasesGrid.innerHTML = '';
    
    diseases.forEach(disease => {
        const diseaseCard = document.createElement('div');
        diseaseCard.className = 'disease-card';
        diseaseCard.onclick = () => showDiseaseModal(disease.name, disease);
        
        const specialists = disease.specialists || ['Терапевт'];
        const specialistsHtml = specialists.slice(0, 2).map(spec => 
            `<span class="specialist-mini-tag">${spec}</span>`
        ).join('');
        
        diseaseCard.innerHTML = `
            <h3>${disease.name}</h3>
            <p>${(disease.description || '').substring(0, 100)}...</p>
            <div class="disease-tags">
                <span class="disease-tag">${disease.category || 'Общее'}</span>
                ${specialistsHtml}
            </div>
        `;
        
        diseasesGrid.appendChild(diseaseCard);
    });
}

function filterDiseases(category) {
    const filterButtons = document.querySelectorAll('.filter-btn');
    filterButtons.forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    loadDiseases();
}

function searchDiseases() {
    const searchInput = document.getElementById('diseaseSearch');
    const searchTerm = searchInput ? searchInput.value.trim() : '';
    
    loadDiseases();
}

// Модальное окно заболевания
function showDiseaseModal(name, info) {
    const modal = document.getElementById('diseaseModal');
    const diseaseName = document.getElementById('modalDiseaseName');
    const diseaseContent = document.getElementById('modalDiseaseContent');
    
    if (!modal || !diseaseName || !diseaseContent) return;
    
    diseaseName.textContent = name;
    
    // Специалисты
    const specialists = info.specialists || ['Терапевт'];
    const specialistsHtml = specialists.map(spec => 
        `<span class="specialist-tag" onclick="findHospitalsBySpecialist('${spec}')">
            <i class="fas fa-user-md"></i> ${spec}
        </span>`
    ).join('');
    
    diseaseContent.innerHTML = `
        <div class="disease-info">
            <div class="info-section">
                <h3><i class="fas fa-info-circle"></i> Описание</h3>
                <p>${info.description || 'Нет описания'}</p>
            </div>
            
            ${info.symptoms ? `
            <div class="info-section">
                <h3><i class="fas fa-exclamation-circle"></i> Симптомы</h3>
                <ul>
                    ${info.symptoms.map(s => `<li>${s}</li>`).join('')}
                </ul>
            </div>
            ` : ''}
            
            ${info.causes ? `
            <div class="info-section">
                <h3><i class="fas fa-search"></i> Причины</h3>
                <ul>
                    ${info.causes.map(c => `<li>${c}</li>`).join('')}
                </ul>
            </div>
            ` : ''}
            
            ${info.complications ? `
            <div class="info-section">
                <h3><i class="fas fa-shield-alt"></i> Осложнения</h3>
                <ul>
                    ${info.complications.map(c => `<li>${c}</li>`).join('')}
                </ul>
            </div>
            ` : ''}
            
            <div class="info-section specialists-modal">
                <h3><i class="fas fa-user-md"></i> К кому обратиться</h3>
                <div class="specialists-list">
                    ${specialistsHtml}
                </div>
                <button class="btn btn-small" onclick="findHospitalsForDisease('${name}')">
                    <i class="fas fa-hospital"></i> Найти больницы
                </button>
            </div>
            
            ${info.when_to_see_doctor ? `
            <div class="info-section warning">
                <h3><i class="fas fa-exclamation-triangle"></i> Когда обязательно обратиться к врачу</h3>
                <p>${info.when_to_see_doctor}</p>
            </div>
            ` : ''}
            
            ${info.ai_usage ? `
            <div class="info-section">
                <h3><i class="fas fa-robot"></i> Использование в ИИ-анализе</h3>
                <p>${info.ai_usage}</p>
            </div>
            ` : ''}
        </div>
    `;
    
    modal.style.display = 'flex';
}

function closeModal() {
    const modal = document.getElementById('diseaseModal');
    if (modal) modal.style.display = 'none';
}

// История анализов
async function loadHistory() {
    const historyList = document.getElementById('historyList');
    if (!historyList) return;
    
    try {
        const response = await fetch('/api/history');
        if (!response.ok) throw new Error('Network error');
        
        const data = await response.json();
        
        if (!data.history || data.history.length === 0) {
            historyList.innerHTML = `
                <div class="empty-history">
                    <i class="fas fa-history"></i>
                    <h3>История анализов пуста</h3>
                    <p>Выполните анализ, чтобы увидеть результаты здесь</p>
                    <button class="btn btn-primary" onclick="showPage('scanner')">
                        <i class="fas fa-camera"></i> Выполнить анализ
                    </button>
                </div>
            `;
            return;
        }
        
        historyList.innerHTML = '';
        
        // Показываем анализы в обратном порядке (последние первыми)
        data.history.reverse().forEach(analysis => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            
            const date = new Date(analysis.timestamp);
            const diseaseName = analysis.result?.disease?.name || 'Анализ не завершен';
            
            historyItem.innerHTML = `
                <div class="history-info">
                    <div class="history-date">
                        ${date.toLocaleDateString('ru-RU')} ${date.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' })}
                    </div>
                    <div class="history-type">
                        ${analysis.type?.charAt(0).toUpperCase() + analysis.type?.slice(1) || 'Неизвестный'} анализ
                    </div>
                    <div class="history-result">
                        Результат: ${diseaseName}
                    </div>
                </div>
                <div class="history-actions">
                    <button class="btn btn-small" onclick="downloadHistoryPDF(${analysis.id})">
                        <i class="fas fa-file-pdf"></i> PDF
                    </button>
                    <button class="btn btn-small btn-outline" onclick="reuseAnalysis(${analysis.id})">
                        <i class="fas fa-redo"></i> Повторить
                    </button>
                </div>
            `;
            
            historyList.appendChild(historyItem);
        });
    } catch (error) {
        console.error('Error loading history:', error);
        historyList.innerHTML = '<p class="error">Ошибка загрузки истории</p>';
    }
}

async function downloadHistoryPDF(analysisId) {
    try {
        const response = await fetch('/api/generate-pdf', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                analysis_id: analysisId
            })
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `medical_report_${analysisId}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } else {
            alert('Ошибка при скачивании PDF');
        }
    } catch (error) {
        alert('Ошибка: ' + error.message);
    }
}

function reuseAnalysis(analysisId) {
    alert('Функция повторного анализа в разработке. Для повторного анализа загрузите новое изображение.');
    showPage('scanner');
}

async function clearHistory() {
    if (confirm('Вы уверены, что хотите очистить историю анализов? Это действие нельзя отменить.')) {
        try {
            const response = await fetch('/api/clear-history', {
                method: 'POST'
            });
            
            if (response.ok) {
                loadHistory();
                alert('История очищена');
            } else {
                throw new Error('Ошибка сервера');
            }
        } catch (error) {
            alert('Ошибка при очистке истории: ' + error.message);
        }
    }
}

// Вспомогательные функции для больниц (заглушки)
function findHospitalsBySpecialist(specialist) {
    alert(`Поиск больниц со специалистом: ${specialist}\n\nЭта функция в разработке. В реальном приложении здесь будет интеграция с картами и базами данных медицинских учреждений.`);
}

function findHospitalsForDisease(disease) {
    alert(`Поиск больниц для лечения: ${disease}\n\nЭта функция в разработке. В реальном приложении здесь будет интеллектуальный поиск медицинских учреждений по специализации.`);
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    // Настройка drag & drop
    const uploadArea = document.getElementById('uploadArea');
    
    if (uploadArea) {
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.style.borderColor = '#4f46e5';
            this.style.backgroundColor = '#c7d2fe';
        });
        
        uploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.style.borderColor = '#e5e7eb';
            this.style.backgroundColor = 'transparent';
        });
        
        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.style.borderColor = '#e5e7eb';
            this.style.backgroundColor = 'transparent';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (file.type.match('image.*')) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        showPreview(e.target.result);
                    };
                    reader.readAsDataURL(file);
                } else {
                    alert('Пожалуйста, выберите файл изображения');
                }
            }
        });
    }
    
    // Выбор типа анализа
    document.querySelectorAll('.type-option').forEach(option => {
        option.addEventListener('click', function() {
            document.querySelectorAll('.type-option').forEach(opt => {
                opt.classList.remove('active');
            });
            this.classList.add('active');
        });
    });
    
    // Закрытие модальных окон при клике вне их
    window.addEventListener('click', function(event) {
        const modal = document.getElementById('diseaseModal');
        const cameraModal = document.getElementById('cameraModal');
        
        if (event.target === modal) {
            modal.style.display = 'none';
        }
        
        if (event.target === cameraModal) {
            closeCamera();
        }
    });
    
    // Обработка клавиши Esc для закрытия модальных окон
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
            closeModal();
            closeCamera();
        }
    });
    
    // Загрузка истории чата
    loadChatHistory();
    
    // Если история чата пуста, добавляем приветственное сообщение
    if (chatHistory.length === 0) {
        setTimeout(() => {
            addChatMessage('Привет! Я ваш медицинский ассистент. Задайте мне вопрос о симптомах, заболеваниях или здоровье. Помните, что я не заменяю врача, а лишь предоставляю информацию.', 'ai');
        }, 500);
    }
    
    // Начальная загрузка заболеваний
    setTimeout(loadDiseases, 100);
});