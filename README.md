# SyncGraph 309

## Table of Contents

- [SyncGraph 309](#syncgraph-309)
  - [Table of Contents](#table-of-contents)
  - [About this Project](#about-this-project)
  - [Installation \& Usage](#installation--usage)
    - [Prerequisites](#prerequisites)
    - [Running the Application](#running-the-application)
  - [File Content and Structure](#file-content-and-structure)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)
  - [Acknowledgements](#acknowledgements)

## About this Project

SyncGraph 309 is a project that leverages OpenRouter, OpenAI, Streamlit, and various ML techniques with Sklearn to create predictive models from 
datasets that can be found within the Toronto Police Service's [Public Data Portal](https://data.torontopolice.on.ca/)

## Installation & Usage

### Prerequisites

For this project, it is recommended that you have Python 3.8 or higher installed on your system, alongside Pipenv or Poetry (whichever package manager you prefer). Additionally, you will need to have Git installed on your system, with Streamlit installed as well since our Steamlit application will need to use the command-line interface to run the application properly.

### Running the Application
1. Clone the repository from GitHub to your local machine:
```bash
git clone https://github.com/Hi-kue/syncgraph-309.git
```
2. Open 2 terminals, navigating to the `client` directory and `server` directory respectively and running the following commands:
```bash
cd ./client/
pipenv install
pipenv shell

cd ./server/
pipenv install
pipenv shell 
```

> [!NOTE]
> Alternatively, we suggest to use a single package manager with a single virtual environment for both the client and server, but if you prefer to have separate environments for each, you can do so as well.

3. Once both the client and the server have been set up, you should start with the `server` directory and run the ipynb file called `c309_r2_toodu_model.ipynb` to train and deploy the `.pkl` model files that will be used in the Streamlit and backend application.

> [!NOTE]
> In order to quickly run the cells within the notebook, you can use press `SHIFT + ENTER` for each cell to run them.

4. Once the models have been trained and deployed and the `.pkl` files have been saved, you can now run the `app.py` file in the server by either running the bash script `./run.sh` or by typing in `flask run` in your first terminal.

5. Finally, after running the `app.py` file, you can now run the Streamlit application (on the other terminal) by typing in the following command or by running the bash script `./run_streamlit.sh`:
```bash
streamlit run app_streamlit.py --server.port 8501
```
> [!NOTE]
> For more information about how to run the application properly, consider viewing the youtube video tutorial that will be attached to this README file below.

If you are still having trouble running the application, consider watching the video tutorial below:

> [!YOUTUBE VIDEO]()

## File Content and Structure

Provided below are the files and folders for this project in a tree structure:
```plaintext
.
├── client/
│   ├── main.py -> (StreamLit App)
│   ├── host.sh
│   ├── run.sh
│   ├── .env
│   ├── .env.sample
│   ├── poetry.lock
│   └── pyproject.toml
├── server/
│   ├── data/
│   │   └── *.{csv, txt, xlsx}
│   ├── models/
│   │   └── *.pkl
|   ├── insights/
|   │   └── *.ipynb
│   ├── app.py
│   ├── pyproject.toml
│   ├── poetry.lock
│   └── setup.sh
├── docs/
│   ├── CODE_OF_CONDUCT.md
│   ├── CODEOWNERS
│   ├── CONTRIBUTING.md
│   └── SECURITY.md
├── .gitignore
├── README.md
└── LICENSE
```

## Contributing

Contributions are what make open-source such an amazing place to learn, get inspired, and create. Any contributions you make to this repository is **greatly appreciated**.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions or concerns, feel free to reach out to the contributors of this website or open an issue in this repository following the proper guidelines in the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## Acknowledgements

Thank you to all the following people for their contributions to this repository, to the moon and back! 🚀

**TODO**: Add contributors here (if any).