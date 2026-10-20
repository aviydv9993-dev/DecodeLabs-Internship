# Rule-Based AI Chatbot ("Byte")

**DecodeLabs Industrial Training Kit - Artificial Intelligence, Project 1 (Batch 2026)**

A Python command-line chatbot that replies to predefined user inputs using rules and dictionaries. No machine learning is used. Every reply is traceable from input to logic to output.

## Features
- Continuous `while True` loop that runs until the user types an exit command
- Input sanitization: lowercase, trimmed spaces, punctuation removed
- Knowledge base: dictionary with 9 intents (greeting, how are you, name, help, joke, tip, thanks, time, date)
- Keyword lookup table (dictionary) instead of a long `if-elif` ladder
- Fallback reply for anything it does not understand
- Remembers your name (`my name is ...`)
- Random reply variety and live time/date answers
- Clean exit with `bye`, `exit` or `quit`

## How to run
```bash
python chatbot.py
```
Requires Python 3.9+. No external libraries needed.

## Example
```
Byte: Hi! I'm Byte. Type 'help' for ideas or 'bye' to exit.
You: Hello!
Byte: Hey there! I'm Byte, your coding study buddy.
You: my name is Riya
Byte: Nice to meet you, Riya!
You: tell me a joke
Byte: Why do programmers prefer dark mode? Because light attracts bugs!
You: bye
Byte: Goodbye, Riya! Keep coding.
```

## How it works (IPO model)
1. **Input:** read the message and clean it.
2. **Process:** look up the intent in a dictionary (O(1) lookup).
3. **Output:** print a reply from the knowledge base, or a fallback.

## Skills demonstrated
Control flow, loops, dictionaries, string handling, decision-making logic, basic AI concepts.

## Author
Your Name - Avinash,DecodeLabs AI Intern, Batch 2026
