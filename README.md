# Stupid AI Coder

Stupid AI Coder is a rather silly Python script that uses AI to make and improve code projects automatically. Here's how it works in simple terms:

![outputFast](https://github.com/user-attachments/assets/f9b96140-ce48-4462-8962-adf982953f36)

1. You tell it what kind of program you want to make. Don't limit yourself, get creative! Use any language you like.
2. You pick which AI "brain" to use (like Claude, GPT-4o-mini, or local Llama3.2).
3. The AI writes some basic boilerplate code to start with.
4. Then, it _keeps trying_ to make the code better, over and over again. Each iteration adds one new feature.
5. It checks if the final version of the code works without errors. (This bit is tricky and not always working.)
6. Finally, it puts together a whole project folder with all the files you need.

The script does this by talking to AI language models, which are like really smart computer brains that can understand and write code. It asks the AI to write code, then to improve it, and keeps doing this until it has a working program. It also creates helpful files like a README and a list of things you need to install to run the program.

## Features

- Multiple AI model support (Claude, GPT-4, Llama)
- Iterative code improvement with diff visualization
- Automatic error detection and correction
- Project folder structure generation
- README and requirements.txt file creation
- Colorful console output with progress indicators
- Option to input task description from a file

## Usage

1. Ensure you have Python 3.7+ installed.

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the script:
   ```
   python ai-coder.py
   ```
   
   Or, to use a file as input for the task description:
   ```
   python ai-coder.py -f path/to/your/task_description.txt
   ```

4. Follow the prompts:
   - Enter your coding task description (if not using a file input)
   - Choose the number of iterations (1-20)
   - Select an AI model

5. The script will generate your project in a new folder named after the auto-generated project title.

## Output

The script generates a project folder containing:
![CleanShot 2024-10-07 at 23 32 12@2x](https://github.com/user-attachments/assets/46f0f59f-6e97-4d99-ab5b-c1f0bd8d7909)

- Final code file
- README.md
- requirements.txt
- .gitignore
- A 'versions' subfolder with intermediate iterations

## Customization

You can modify the script to adjust:

- Available AI models
- Maximum iterations
- Error handling attempts
- Logging verbosity

## Notes

- The quality of the generated code depends on the chosen AI model and the clarity of the task description.
- Always review and test the generated code before using it in production environments.
- The script requires an active internet connection to communicate with AI models.
- When using a file input, ensure the task description is clear and detailed for best results.

## Contributing

Contributions to improve AI Coder are welcome! Please submit issues and pull requests on the project's GitHub repository.
