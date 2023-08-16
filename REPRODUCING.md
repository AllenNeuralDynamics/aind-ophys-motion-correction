This [Code Ocean](https://codeocean.com) Compute Capsule will allow you to run and reproduce the results of [aind-ophys-suite2p-motion-correction](https://codeocean.allenneuraldynamics.org/capsule/7296765/tree) on your local machine<sup>1</sup>. Follow the instructions below, or consult [our knowledge base](https://help.codeocean.com/user-manual/sharing-and-finding-published-capsules/exporting-capsules-and-reproducing-results-on-your-local-machine) for more information. Don't hesitate to reach out via live chat or [email](mailto:support@codeocean.com) if you have any questions.

<sup>1</sup> You may need access to additional hardware and/or software licenses.

# Prerequisites

- [Docker Community Edition (CE)](https://www.docker.com/community-edition)

# Instructions

## Download attached Data Assets

In order to fetch the Data Asset(s) this Capsule depends on, download them into the Capsule's `data` folder:
* [Other_667826_2023-04-10_16-08-00](https://codeocean.allenneuraldynamics.org/data-assets/00bc83eb-6496-42f5-87c6-f0202e125eda) should be downloaded to `data/Other_667826_2023-04-10_16-08-00`

## Log in to the Docker registry

In your terminal, execute the following command, providing your password or API key when prompted for it:
```shell
docker login -u ariellel@alleninstitute.org registry.codeocean.allenneuraldynamics.org
```

## Run the Capsule to reproduce the results

In your terminal, navigate to the folder where you've extracted the Capsule and execute the following command, adjusting parameters as needed:
```shell
docker run --platform linux/amd64 --rm \
  --workdir /code \
  --volume "$PWD/code":/code \
  --volume "$PWD/data":/data \
  --volume "$PWD/results":/results \
  registry.codeocean.allenneuraldynamics.org/capsule/284b5530-c9a4-4ad7-b19e-3e964facbdda \
  bash run
```
