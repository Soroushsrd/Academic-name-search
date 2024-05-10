# Academic-name-search

This Python script utilizes the Tavily, LangChain, and OpenAI GPT-4 libraries to search for academic professors and researchers. It outputs their name, recent background and research, their position, and email.

## Dependencies

- tavily - Python client for Tavily API.
- langchain_openai - Library for OpenAI GPT-4 integration.
- langchain_community - Additional modules for LangChain.
- streamlit - Framework for building interactive web applications.
- requests - Library for making HTTP requests.
- bs4 (Beautiful Soup) - Library for web scraping.

To run the application locally, follow these steps:

- Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```
- Obtain an API key for OpenAI and set it as an environment variable named OPENAI_API_KEY.
- Obtain an API key for Tavily and replace 'Your_API' in the script with your API key.

```toml
[streamlit]
OPENAI_API_KEY = "your_openai_api_key"
```


## Usage


1. Run the Streamlit app:
```bash
streamlit run app.py
```
2. Enter the name of the professor/Dr you're trying to look up.
3. Optionally, add another search query.
4. Click the "Start" button to initiate the search.

## Example
![Name Search](https://github.com/Soroushsrd/Academic-name-search/blob/main/Screenshot%202024-05-10%20165615.png)

## Contributing

- This application relies on the LangChain library for natural language processing tasks.
- The GPT-4 Turbo model is provided by OpenAI.
- Tavily is used to search the internet.

## License

This project is licensed under the [Apache License 2.0](http://www.apache.org/licenses/)
