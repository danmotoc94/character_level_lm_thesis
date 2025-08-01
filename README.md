Character-level Language Model for Name Generation
🎓 Bachelor's Thesis in Computer Science

Author: Motoc Dan
University: UAV Arad
Department: Computer Science
Project Description

This repository contains the code and resources for a bachelor's thesis that explores building a character-level language model. The goal of this project is to generate new names by learning the statistical patterns of character sequences from a given dataset.

The project implements and compares two key approaches:

    A traditional bigram statistical model: This approach calculates the probability of one character following another and uses this to generate new words.

    A neural network implementation using PyTorch: This more modern approach uses a simple neural network to learn the same character transition probabilities. The model is trained using a process called gradient descent, where the weights are iteratively adjusted to minimize the negative log-likelihood loss, thereby improving the model's ability to predict the next character in a sequence. This provides a foundation for understanding more complex models like recurrent neural networks (RNNs).

The accompanying code is a well-commented script that demonstrates both methods, from data preparation to name generation and model evaluation.
Technologies and Libraries

This project was developed using the following key technologies and libraries:

    Python 3.8+: The primary programming language.

    PyTorch: A powerful open-source machine learning framework used for the neural network implementation and its gradient-based optimization.

    Matplotlib: A plotting library used for visualizing the bigram counts.

Installation and Usage

To get this project up and running on your local machine, follow these steps:
1. Clone the Repository

First, clone this repository to your local machine.


2. Prepare the Dataset

The project requires a simple text file named names.txt containing one name per line. If you don't have one, you can use the following command to create a placeholder:

echo "emma\nemma\njohn\nmaria\nalex\nsarah" > names.txt


3. Set Up the Environment

It's recommended to use a virtual environment to manage dependencies.

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS and Linux
source venv/bin/activate
# On Windows
venv\Scripts\activate


4. Install Dependencies

Install all the required Python libraries using pip:

pip install torch matplotlib


5. Run the Code

The generate_names.py script will train both the statistical bigram model and the PyTorch neural network, then generate sample names.

python generate_names.py


Accessing the Thesis Document

The complete bachelor's thesis document, which provides a detailed theoretical background, methodology, results, and discussion, is available here:

    [Link to your final thesis PDF, e.g., in this repository or on your university's site]

License

This project is licensed under the MIT License - see the LICENSE file for details.