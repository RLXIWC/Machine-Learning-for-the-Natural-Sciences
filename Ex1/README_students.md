# Introduction to AI

The exercise sheets will be written in the form of Jupyter Notebooks and handed out as .ipynb
files. We use Docker to ensure that we have a uniform environment across all devices and
operating systems.

# Installation instructions

The installed software, its dependencies as well as data used throughout
the course require some space.
Make sure to have at least 20GB of disk space. The container is quite large since
it contains everything we need for all upcoming assignments.

1. Install [Docker](https://docs.docker.com/get-docker/). Docker commands
   may require you to run commands as admin/root.
2. Test your Docker installation by running ``docker run hello-world``. Depending on your
   configuration, you may need to manually start the docker daemon after the first installation.
   You can check whether docker runs at all with `docker ps`, `docker images` or `docker info`. If
   they have the same result, then Docker is not running. The easiest way is to configure docker to
   start on boot and then reboot.
3. The command below starts your container and maps your current working directory as a directory.
   This way, you can run it from any directory you like.
4. ``cd`` into the directory with your downloaded assignment. Then run
   ```
   # Linux and Mac
   docker run -it -p 8888:8888 -v "`pwd`:/auto-grading/assignments" --rm --name "aimat" "patchmeifyoucan/aimat-ml4nat:1.0.0" "poetry run jupyter notebook --allow-root --no-browser --ip 0.0.0.0"
   
   # Windows
   docker run -it -p 8888:8888 -v "%cd%:/auto-grading/assignments" --rm --name "aimat" "patchmeifyoucan/aimat-ml4nat:1.0.0" "poetry run jupyter notebook --allow-root --no-browser --ip 0.0.0.0"
   ``` 
5. Docker will download the container image and start Jupyter in a container. Doing so might take some
   time as everything is included such that there is only one dedicated programming and grading environment.
6. Open the URL pointing to `http://127.0.0.1:8888?token=` at the bottom of the console output. Not any of the other
   links.

# Submission Instructions

Please follow these steps precisely. <span style="color:red">**There are no optional steps here.**</span>
It is an automated environment, so any modification of the workflow might result in an invalid grading file.
This way, the grader is not able to do some of its magic and hence will reject your result.
The result is 0 points in most cases.
Also, please ask questions on the board such that they are visible for everybody. In most cases there
is no problem with the grader.

1. Download the submission files from Ilias.

2. <span style="color:red">**It may happen that a new container version was released.**</span>
   Each notebook must be edited using the container version it was generated with. Otherwise,
   the environment may have wrong or missing dependencies. Find the current container version at the
   top of each assignment and change the start command accordingly, i.e., replace ``1.0.0``
   with the correct version.
3. Start Jupyter from Docker and edit your notebook.

4. <span style="color:red">**Validate your solution**</span>: Assignment notebooks have inline
   tests. They are special cells in blue with assert statements and should not be changed and allow
   you to validate your submission locally prior to submission. These tests are not part of the
   grading process and are they there to give you an easy way to check whether your submission is
   working correctly in principle. Hence, passing the validation is not directly equivalent to
   receiving points during grading, but it is a necessary requirement.

5. Upload your solution. Once you have finished your solution, upload the .ipynb notebook to Ilias.
   In general, there should be exactly <span style="color:red">**ONE**</span> file you upload. Ilias
   will create one archive with all the submissions in a fixed directory structure. As this is the basis
   for grading, submissions not meeting this requirement won't be graded (0 points).

6. As a summary, do <span style="color:red">**NOT**</span> do any of the following:
    - <span style="color:red">**DO NOT**</span> use the ZIP upload. Instead, upload only the notebook file.
    - <span style="color:red">**DO NOT**</span> upload data. Instead, upload only the notebook file.
    - <span style="color:red">**DO NOT**</span> rename the notebook. No `_final`, `_uxxxx`, `_etc_etc` suffixes.
    - <span style="color:red">**DO NOT**</span> upload multiple versions. Ilias will rename these, violating the above
       requirement.
       If you find yourself needing multiple versions <span style="color:red">**delete the
       old one first.**</span>

7. After grading, you will receive task feedback via email.

# Why we use Docker

1. We are often confronted with the question "whether the grader works" (TL;DR: it does).
2. We would like to help in cases where the grading results are poor, even though unexpected by students.

3. Top 3 reasons for unexpected grading results.
    - Students don't read the setup instructions properly.
    - Students have broken computers (the weirdest cases include the Windows system folder not being in PATH, which
      causes ~10000 different errors).
    - Students do not follow any instructions at all, providing their own solution.

4. Providing installation scripts requires everybody to set up everything precisely.
    - In practice, fails due to the above reasons.

5. We cannot help with problems we do not control.
    - Which OS you use.
    - Which library versions you use.
    - Where you write code.

6. To provide a proper working environment for you and us, we provide a Docker container.

7. Benefits
   - No setup, except installing Docker. If that does not work, the grading likely wouldn't work either.
   - Fixed environment for you and us. What works keeps working. Something that does not work can be fixed once and
     applied to everybody.
   - Does not run into OS-dependent issues (we even support the HML distro, look it up).
   - Avoids the "it works on my machine" problem
