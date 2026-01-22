# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

`![my_image](figures/<image>.<extension>)`

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

or

```bash
uv add typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [ ] Make sure that all team members have write access to the GitHub repository (M5)
* [ ] Create a dedicated environment for you project to keep track of your packages (M2)
* [ ] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [ ] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [ ] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [ ] Remember to either fill out the `requirements.txt`/`requirements_dev.txt` files or keeping your
    `pyproject.toml`/`uv.lock` up-to-date with whatever dependencies that you are using (M2+M6)
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [ ] Do a bit of code typing and remember to document essential parts of your code (M7)
* [ ] Setup version control for your data or part of your data (M8)
* [ ] Add command line interfaces and project commands to your code where it makes sense (M9)
* [ ] Construct one or multiple docker files for your code (M10)
* [ ] Build the docker files locally and make sure they work as intended (M10)
* [ ] Write one or multiple configurations files for your experiments (M11)
* [ ] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [ ] Use logging to log important events in your code (M14)
* [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [ ] Write unit tests related to the data part of your code (M16)
* [ ] Write unit tests related to model construction and or model training (M16)
* [ ] Calculate the code coverage (M16)
* [ ] Get some continuous integration running on the GitHub repository (M17)
* [ ] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [ ] Add a linting step to your continuous integration (M17)
* [ ] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [ ] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [ ] Create a trigger workflow for automatically building your docker images (M21)
* [ ] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [ ] Create a FastAPI application that can do inference using your model (M22)
* [ ] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [ ] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [ ] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Setup collection of input-output data from your deployed application (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [ ] Make sure all group members have an understanding about all parts of the project
* [ ] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

4

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s244705, s245310, s244794, s252362

### Question 3
> **Did you end up using any open-source frameworks/packages not covered in the course during your project? If so**
> **which did you use and how did they help you complete the project?**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We did not use any open-soruce frameworks or packages that were not covered in the course for our project. 

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We used UV for managing our dependencies, because of its increasing relevance in the field. All dependencies are declared in the pyproject.toml file, while exact versions are pinned in a lockfile generated by uv. This lockfile ensures reproducibility across machines and development stages.

To obtain a complete copy of our development environment, a new team member would first install uv and ensure a compatible version of python. After cloning the repository, they would run 'uv sync', which automatically creates a virtual environment and installs all dependencies exactly as specified in the lockfile.


### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

We initialized our project using the course-provided Cookiecutter MLOps template, which served as structured starting point for organizing our codebase. From the template, we primarily filled out source directory ('src'), chich containts the core implementation of our project. This folder represents the main workflow of our machine learning pipeline.

The data directory does store contain raw datasets directly in the repository. Instead, raw data is sourced programmatically from kaggle and tracked using DVC. The .dvc directory contains metadata files that refrence versioned data stored externally.

To improve configuration and reproducability, we added a config directory for managing model and pipeline parameters, as well as .devcontainer directory. We also removed the notebooks directory, as our workflow relied on script-based development rather than interactive notebooks. 

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

Our code is formatted after the official Python style guide, Pep8. We added 'Ruff' as a development dependency to enforce consistent code formating, structure, and linting, healping us maintain clean and readable code. In addition, we sometimes used type hints to improve code reliability and make function inputs and outputs easier to understand. We also included documentaiton and comments where to appropriately clarify the purpose of components.

While good coding practices may seem less important in solo projects, they become essential when dealing with larger projects with multiple contributers. Consistent formatting makes the codebase easier to read, typing helps prevent bugs and misunderstandings, and documentation ensures other developers can quickly understand and expand the project. 

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer: 

In total we have implemented 5 tests, unit tests namely data and model testing as well as integration and performance tests for the backend and frontend of our API. During the data and model tests we ensure the data is properly split, is in the right shape, the distinct models are accurately trained and appear in a structure which is reasonable, further whether our output has the desired shape. For our API we test how its performance is effected under several users by locust, and whether in the backend the model directories can be found as well as prediction is accuractely done, and in the frontend simply whether there is proper connection with the backend.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

We get a total code coverage of 28% which is not necessarily high, but this also is grounded in the fact that we were unable to remove cloud_train and old_cloud_train files from also being in the coverage despite the absence of any tests for these two files, which brings the overall coverage significantly down. Even if we had 100% coverage, it doesnt mean the code is error free as coverage merely looks at the amount of code that is tested, but doesnt give any indication in regards to whether these tests work accurately, as well as not including edge cases, such as empty inputs and the competency of several files together for the most part, while we do check frontend backend competency. 

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We made use of both branches and pull requests. Everyone would create their own branch when they were working on a new objective, named after the objective for easier tracking of the branches and their commits, and either when it was completely done or in an intermediate process of the objective, the branch would be merged into the main. This was mainly to ensure that neither of the branches were too far behing commits of the main as we faced this issue on the first day making merging were cumbersome. Therefore, there would be sent pull request to the remote repository where someone would have to confirm the merge and resolve any conflicts before committing the merge. Also this made it easy to go back and forth between commits to ensure version control, if we had to go back to a previous version. Afterwards the branch would be deleted, just to have a proper organisation of the directory.



### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We used DVC fundamentally as the although we had raw data in our repository for the purpose of creating dockerfiles, since our data was sourced from kaggle and versioned using dvc with metadata stored in the external storage through the .dvc. By this we were able to track the data versions, making sure model training was consistent. And it would normally increase the storage relief as the large dataset we had wouldnt have to remain locally however we had to store data locally due to other problems, this also applies to the members being able to access the same data set without constantly downloading the newest version. We also didnt end up making too many variations of the data that would benefit the logging of training for different dataset, which in a real-life project would prove to be very beneficial.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

We have organized our continous integration into 8 separate files, pre-commit, codecheck, coverage-test,data-change,linting,model-registry, and also we included dependabot. These workflows would be activated in every pull and push request:
Pre-commit: Ruff to get the code quality and format to a certain standard, while ignoring certain of the data directories locally
Code-check: Does a similar job to pre-commit such as formatting the code to a certain standard however manages this between pull requests, safety net for if pre-commit is skipped
Data-change: Executed if there is change in the data quality or situation hence updates the models if there is any change purely in the data/ folder
Model-change: Similarly is only executed when there is changes to the model structures, handles the deployment of the new model.
Linting: Executes the code standard and ruff and formatting similar to code-check between pull and push requests however is extended to work in development branches while code-check is mainly between main and master
Coverage Test: Executes the coverage report of the tests on the following files we have, reporting it.
Dependabot: Runs dependency checks automatically and updates them
For all of the workflows we either use python 3.12 or 3.11 for the best competency, we run on ubuntu-latest however we dont test several operating systems. We use action cacheing that reduces our runtime. Here is a github action that includes coverage test: https://github.com/WilliamKentoRasmussen/MLOPS-PROJECT/actions/runs/21244110734

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We configured training experiments using Hydra configuration files to separate model, training, and system settings. The config directory contains subfolders for model (e.g. Baseline, Alexnet, VGG16) and training (e.g., default and quick runs), each defining parameters such as architecture, learning rate, batch size, and epochs. A central config.yaml composes these components and specifies data paths, device selection, and output directories. Experiments are executed by selecting configurations at runtime, for example: 
```bash
python src/main_project/train.py model=baseline training=default
```
Hydra also allows for hyperparameters to be overridden from the command line, for example: 
```bash
python src/main_project/train.py training.epochs=20 training.lr=1e-4
```

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

We ensured experiment reproducibility by combining configuration management, data versioning, and environment control. All experiments are configured using Hydra configuration files, which separate model, training, and system parameters. Whenever an experiment is run, Hydra automatically saves the fully resolved configuration for that run, ensuring that no parameter choices are lost and that each experiment can be tracked back to its exact settings. 

Data reproducibility is handled using DVC. Rather than storing raw data in the repository we track data version through DVC metadata files, allowing experiment to be rerun with the same data version even as datasets evolve. Dependency reproducibility is ensured using uv, Where all python dependencies are declared and pinned in a lockfile committed to version control.

To reproduce and experiment, one would restore the correct data using DVC, synchronize dependencies using uv, and rerun the training script with the same hydroconfiguration. Additionally, a devcontainer/Docker setup ensures a consistent development environment across machines, further reducing sources of variability.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

![my_image](figures/WandB_1.png)
![my_image](figures/WandB_2.png)

We used Weights & Biases (W&B) to track and compare our machine learning experiments across different model architectures and training configurations. Each run corresponds to a specific model type allowing us to systematically evaluate perfromance differences. 

As seen in the images above, we have tracked epochs, best validation accuracy, training loss, training accuracy, validation loss, and validation accuracy. Training metrics inform us about how well the model fits the training data, while validation metrics are critical for detecting overfitting and assessing generalization performance. In particular, validation accuracy serves as our primary model selection criterion, as it reflects perfromance on unseen data. we also logged best validaiton accuracy to esily compare the peak perfromance of the best runs. Additionally, we tracked epoch count to align metric across runs and ensure fair compairson when early stopping was triggered 

Furthermore, System metrics were also tracked. These metrics are important from an MLOps perspective, as they provide insight into hardware efficiency, resource consumption, and potential bottlenecks during training. 

By visualizing all metrics in W&B, we were able to compare models such as the baseline, AlexNet, and VGG16 side by side. THis enabled systematic comparison of model behavior and performance across experiments. It should be noted that the reported results do not reflect fully optimized hyperparameters, as the priority focus of this work was on implementing a reproducible and well-intrumented MLOps pipeline rather than maximizing model performance.  

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

We used docker to significantly increase our training time and the reproducibility of the training with the exact same dataset, models, dependencies and making it identically reproducable to reach the same results, if deterministic obviously. We had a default config file which the training dockerfile inherited hence we would need to run with docker run --name experiment_new, and the following docker files with the needed configuratons and the training and validation accuracies were logged into wandb. We also used docker to train in the cloud as well as running the backend and frontend dockerfiles, hence we had 4 different images for each of the purposes. Here is the output from the training image, was a bit unsure as to how to share a dockerfile.

\--env-file .env \train:new docker run --hostname=f3376763256a --env=WANDB_API_KEY=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 --env=WANDB_PROJECT=final_assignment_mlops --env=WANDB_ENTITY=s244794-danmarks-tekniske-universitet-dtu --env=PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin --env=LANG=C.UTF-8 --env=GPG_KEY=7169605F62C751356D054A26A821E680E5FA6305 --env=PYTHON_VERSION=3.12.12 --env=PYTHON_SHA256=fb85a13414b028c49ba18bbd523c2d055a30b56b18b92ce454ea2c51edc656c4 --env=UV_TOOL_BIN_DIR=/usr/local/bin --volume=/Users/keremozemre/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/MLOPS_Project/MLOPS-PROJECT/data/processed:/app/data/processed --network=bridge --restart=no --label='org.opencontainers.image.created=2026-01-15T20:29:43.472Z' --label='org.opencontainers.image.description=An extremely fast Python package and project manager, written in Rust.' --label='org.opencontainers.image.licenses=Apache-2.0' --label='org.opencontainers.image.revision=ee4f0036283a350681a618176484df6bcde27507' --label='org.opencontainers.image.source=https://github.com/astral-sh/uv' --label='org.opencontainers.image.title=uv' --label='org.opencontainers.image.url=https://github.com/astral-sh/uv' --label='org.opencontainers.image.version=0.9.26-python3.12-bookworm' --runtime=runc -d train:new

   Building main-project @ file:///

      Built main-project @ file:///

Uninstalled 1 package in 2ms

Installed 1 package in 0.49ms

wandb: Currently logged in as: s244794 (s244794-danmarks-tekniske-universitet-dtu) to https://api.wandb.ai.⁠ Use `wandb login --relogin` to force relogin

wandb: setting up run wi5ewrip

wandb: Tracking run with wandb version 0.23.1

wandb: Run data is saved locally in /wandb/run-20260116_101652-wi5ewrip

wandb: Run `wandb offline` to turn off syncing.

wandb: Syncing run baseline_0.001_32

wandb: ⭐️ View project at https://wandb.ai/s244794-danmarks-tekniske-universitet-dtu/chest-xray-classification⁠

wandb: 🚀 View run at https://wandb.ai/s244794-danmarks-tekniske-universitet-dtu/chest-xray-classification/runs/wi5ewrip⁠

wandb: WARNING Symlinked 1 file into the W&B run directory; call wandb.save again to sync new files.

wandb: updating run metadata

wandb: uploading models/baseline_best.pth; uploading output.log; uploading wandb-summary.json

wandb: uploading models/baseline_best.pth; uploading config.yaml

wandb: uploading history steps 0-1, summary, console lines 10-16

wandb: 

wandb: Run history:

wandb: best_val_accuracy ▁

wandb:             epoch ▁

wandb:    train/accuracy ▁

wandb:        train/loss ▁

wandb:      val/accuracy ▁

wandb:          val/loss ▁

wandb: 

wandb: Run summary:

wandb: best_val_accuracy 68.75

wandb:             epoch 1

wandb:    train/accuracy 85.62117

wandb:        train/loss 0.32086

wandb:      val/accuracy 68.75



### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

Debugging method was dependent on group member. In general, when encountering bugs during experiment execution, We first attempted to identify the source of the problem by inserting print statements and inspecting intermediate outputs. For more complex or unclear errors, we made use of AI-assisted tools such as GitHub Copilot to help interpret error messages and suggest potential fixes. This iterative approach allowed us to quickly diagnose configuration issues. 

In addition to debugging, we performed basic profiling of the training loop using Pytorch's built-in ```torch.profiler```. The profiler was used to measure CPU execution time during training, helping us identify potential performance bottlenecks in the data loading and forward/backward passes. Profiling results were exported and visualized using 'ui.perfetto.dev'. While the code is not assumed to be fully optimized, this profiling step provided insight into runtime behavior and profiling implementations.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We used the following service: Artifacts, Cloud Build, Cloud Run, and Vertex AI. Artifacts is used for storing our docker images. We implemented an automatic trigger in cloud build to automaticly build images in the cloud and upload them to artifacts. Moreover, we used the Bucket service to store our trained pytorch models as well as the processed pneumonia Data consisting of jpeg images. The cloud run was used to run our backend for our api and Vertex ai was used to train our images.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We used the compute engine to run the training of our models inside the vertex ai. We used the following hardware: CPU and we started the dockerfile on our computer.



### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

![my_image](figures/bucket.png)
![my_image](figures/inside_bucket.png)


### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

![alt text](image-1.png)
[alt text](image.png)

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

--- question 22 fill here ---

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

--- question 23 fill here ---

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

--- question 24 fill here ---

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

We both used unit and load testing for the API. We implemented unit tests for the backend for the point of testing if the endpoints worked accordingly, such as inference of single images as well as whether the model directories were found or not. For the frontend we simply, tested for the backend url properly connecting. We also implemented a load test by locust, where the priority of the tasks was given mainly to single image prediction, we didnt play with load testing too much however the system didn't crash for 10 users, and users per second at 1.

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- question 26 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

--- question 27 fill here ---

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

--- question 28 fill here ---

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 29 fill here ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- question 30 fill here ---

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

--- question 31 fill here ---
