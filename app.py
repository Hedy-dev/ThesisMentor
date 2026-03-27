import streamlit as st
import requests

# Настройка внешнего вида страницы
st.set_page_config(
    page_title="AI Рецензент ВКР",
    # page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL  FastAPI бэкенда
BACKEND_URL = "http://localhost:8000/analyze"

st.title("Проверка структуры ВКР")
st.markdown(
    "Загрузите файл вашей работы в формате `.docx`, чтобы модель-критик проверила "
    "соответствие структуры и содержания практических глав тексту введения."
)

# для отладочной информации
with st.sidebar:
    st.header(" Отладка парсера")
    st.info("структура документа после обработки.")
    debug_container = st.empty()

# 3. Основная область загрузки файла
uploaded_file = st.file_uploader("Выберите файл .docx", type=["docx"])

if uploaded_file is not None:
    # Кнопка для старта, чтобы не отправлять запрос сразу при выборе файла
    if st.button("Запустить анализ", type="primary"):
        
        with st.spinner("Анализируем документ... Это может занять несколько секунд."):
            try:
                # Подготовка файла для отправки через requests
                files = {
                    "file": (
                        uploaded_file.name, 
                        uploaded_file.getvalue(), 
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                }
                
                # Отправка POST-запроса на бэкенд
                response = requests.post(BACKEND_URL, files=files)
                
                # Обработка успешного ответа
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", {})
                    errors = results.get("errors", [])
                    recommendations = results.get("recommendations", [])
                    
                    st.success("Анализ успешно завершен!")
                    
                    # Вывод отладочной информации в сайдбар
                    with debug_container.container():
                        st.metric("Всего узлов в графе", results.get("nodes_count", 0))
                        st.subheader("Распознанные разделы:")
                        for sec in results.get("detected_sections", []):
                            st.text(f"🔹 {sec['title']}")

                    # Разделение экрана на две колонки для вывода Критика и Генератора
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.header("Заключение Критика")
                        if not errors:
                            st.success("Отлично! Структурных ошибок не найдено.")
                        else:
                            for i, err in enumerate(errors):
                                with st.expander(f"Недочет #{i+1} (Узел: {err['node_id'][:8]}...)", expanded=True):
                                    st.error(err['description'])
                                    st.caption(f"Статус: {err['error_status']}")

                    # with col2:
                    #     st.header("Советы Генератора")
                    #     if not recommendations:
                    #         st.info("Рекомендаций пока нет.")
                    #     else:
                    #         for i, rec in enumerate(recommendations):
                    #             # Определяем заголовок: если есть исправленный текст, пишем "Улучшение", если нет — "Совет"
                    #             is_structural = rec.get("is_structural", False)
                    #             header = f"Совет #{i+1} (Структура)" if is_structural else f"Улучшение #{i+1}"

                    #             with st.expander(header, expanded=True):
                    #                 st.write(rec['recommendation']) # Тут будет текст от GigaChat
                    #                 if not is_structural:
                    #                     st.caption("Рекомендуется применить к узлу: " + rec['node_id'][:8])
                    with col2:
                        st.header("Советы LISA AI")
                        if not recommendations:
                            st.info("Загрузите файл для получения рекомендаций.")
                        else:
                            for i, rec in enumerate(recommendations):
                                # Определяем иконку: стройка для структуры, перо для текста
                                is_struct = rec.get("is_structural", False)
                                icon = "🏗️" if is_struct else "✍️"

                                with st.expander(f"{icon} Рекомендация #{i+1}", expanded=True):
                                    text_content = rec['suggestion']

                                    if "Исправленный текст:" in text_content:
                                        # Разделяем совет и сам текст для красоты
                                        parts = text_content.split("Исправленный текст:")
                                        st.markdown(f"**Анализ:** {parts[0].replace('Совет:', '').strip()}")
                                        st.success(f"**Вариант для вставки:**\n\n{parts[1].strip()}")
                                    else:
                                        st.write(text_content)

                                    if rec.get("sources"):
                                        st.caption(f"Источник: {', '.join(rec['sources'])}")
                else:
                    # Обработка HTTP-ошибок от бэкенда
                    st.error(f"Ошибка бэкенда: {response.status_code} - {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error(
                    f"Не удалось подключиться к бэкенду по адресу `{BACKEND_URL}`. "
                    f"Убедитесь, что вы запустили `python main.py` в другом окне терминала."
                )
            except Exception as e:
                st.error(f"Произошла непредвиденная ошибка: {e}")
