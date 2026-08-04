
## Contents

- [1 Description](#1-description)
- [2 Installation](#2-installation)
- [3 Minimal Usage](#3-minimal-usage)
- [4 All Parameters](#4-all-parameters)
- [5 Results (Outputs)](#5-results-outputs)
- [6 Advanced Usage](#6-advanced-usage)
- [7 License](#7-license)

<h1>Feature Selection Pipeline</h1>

<h2 id="desc">1 Description</h2>
<h3>Command-line Docker/Apptainer pipeline</h3>
<p>
  This project provides a unified <strong>feature selection</strong> pipeline that wraps a range of methods
  (<code>GRACES</code>, <code>EAR-FS</code>, <code>CancelOut</code>, <code>STG</code>, <code>CAE</code>,
  <code>DeepLIFT</code>, <code>GradientSHAP</code>, <code>FeatureAblation</code>, <code>Occlusion</code>,
  <code>LIME</code>).
  This standalone command-line pipeline is distributed as a Docker/Apptainer container image and runs locally
  on the user's computer or HPC cluster (e.g., NeSI). It does not require the online web interface or a connection
  to the project server. Users provide local input files, configure the run through CLI arguments, and save outputs
  to mounted host directories.
</p>
<h3>Online web interface</h3>
<p>
  As a separate usage option, an <strong>online web interface</strong> is available at
  <a href="https://server.fs-benchmarking.cloud.edu.au/" target="_blank" rel="noopener noreferrer">server.fs-benchmarking.cloud.edu.au</a>.
  The web interface runs through a browser and does not require users to install Docker or Apptainer locally.
  Users upload a dataset (<code>.npz</code>), select a feature selection method, and configure the available model
  and optimization parameters through the website. Uploaded jobs are processed on the remote server rather than
  on the user's local computer or HPC system. The web interface supports automatic preprocessing and externally
  preprocessed train/test splits, and completed results are returned through the website for download.
</p>
<p class="muted">
  The command-line container pipeline and the online web interface provide two independent ways to access the same
  feature-selection workflow. This repository primarily contains the standalone Docker/Apptainer command-line pipeline
  and its documentation. The online web interface is provided as a separate hosted service and is not the primary
  deployment target of this repository. The sections below therefore document the standalone Docker/Apptainer CLI pipeline
  unless a section explicitly refers to the web interface.
</p>


<h2 id="install">2 Installation</h2>

<h3>2.1 Use the published container images</h3>
<p>
  Two published image variants are available:
</p>
<ul>
  <li>
    <strong>CPU image:</strong> <code>nolanzz/pipeline:latest</code> and <code>nolanzz/pipeline:cpu</code>.
    This is the default image used throughout the examples below and is suitable for local computers and CPU-based HPC jobs.
  </li>
  <li>
    <strong>GPU image:</strong> <code>nolanzz/pipeline:gpu</code>.
    This image is intended for systems with a compatible NVIDIA GPU, NVIDIA driver, and CUDA container support.
  </li>
</ul>
<pre class="card-pre"><code># Apptainer / NeSI: CPU image
apptainer pull pipeline-cpu.sif docker://nolanzz/pipeline:latest

# Apptainer / NeSI: GPU image
apptainer pull pipeline-gpu.sif docker://nolanzz/pipeline:gpu

# Docker: CPU image
docker pull nolanzz/pipeline:latest

# Docker: GPU image
docker pull nolanzz/pipeline:gpu
    </code></pre>
<p class="muted">
  The CPU image is published for both <code>linux/amd64</code> and <code>linux/arm64</code>.
  The GPU image is primarily intended for <code>linux/amd64</code> systems with NVIDIA CUDA support.
</p>

<h3>2.2 Build the Docker image locally from source</h3>
<p>
  The complete source code is available from
  <a href="https://github.com/QINGYUMENG-star/fs_benchmarking" target="_blank" rel="noopener noreferrer">GitHub</a>.
  Docker must be installed and running before building the image.
</p>

<h4>Option A: clone the repository with Git</h4>
<pre class="card-pre"><code>git clone https://github.com/QINGYUMENG-star/fs_benchmarking.git
cd fs_benchmarking

docker build -t fs-pipeline .
docker run --rm fs-pipeline --help
    </code></pre>

<h4>Option B: download and extract the source archive</h4>
<ol>
  <li>Open the GitHub repository and select <strong>Code → Download ZIP</strong>.</li>
  <li>Extract the downloaded ZIP archive.</li>
  <li>Open a terminal and change to the extracted repository root directory containing the <code>Dockerfile</code>.</li>
  <li>Build and test the local Docker image using the commands below.</li>
</ol>
<pre class="card-pre"><code>cd /path/to/fs_benchmarking-main

docker build -t fs-pipeline .
docker run --rm fs-pipeline --help
    </code></pre>

<p class="muted">
  This is a container-based installation, so no separate Python package installation is required on the host.
  The local image name <code>fs-pipeline</code> is only an example and does not require a version tag.
  If you build the image locally, replace <code>nolanzz/pipeline:latest</code> with <code>fs-pipeline</code>
  in the Docker commands shown below. The <code>--help</code> command is a quick smoke test that confirms
  the image starts successfully and exposes the pipeline CLI.
</p>
<p class="muted">
  Available platforms depend on the selected image variant. You can inspect the CPU image manifest with
  <code>docker buildx imagetools inspect nolanzz/pipeline:latest</code> and the GPU image manifest with
  <code>docker buildx imagetools inspect nolanzz/pipeline:gpu</code>.
  Docker and Apptainer will use a compatible image variant when one is available for the host platform.
</p>
<p class="muted">
  The published container image includes the bundled example dataset at
  <code>/app/data/ALLAML_10.npz</code>. Therefore, users can run the bundled example from any host directory
  without cloning the repository or mounting a local input-data directory. Only an output directory needs to be
  mounted so that results persist after the container exits.
</p>
<p class="muted">
  For user-provided datasets, mount the host directory containing the input file to a container path such as
  <code>/data</code>, and mount a separate host directory to <code>/results</code>. Paths supplied to
  <code>--input_path</code> and <code>--out_dir</code> must always use container-side paths.
</p>
<pre class="card-pre"><code># Bundled example with the CPU image: only mount an output directory
mkdir -p result

apptainer run --bind "$(pwd)/result:/results" pipeline-cpu.sif

docker run --rm -v "$PWD/result:/results" nolanzz/pipeline:latest

# Bundled example with the GPU image
apptainer run --nv --bind "$(pwd)/result:/results" pipeline-gpu.sif

docker run --rm --gpus all \
  -v "$PWD/result:/results" \
  nolanzz/pipeline:gpu

# User-provided data with the CPU image: mount both input and output directories
apptainer run \
  --bind "/Users/alice/project/data:/data" \
  --bind "/Users/alice/project/results:/results" \
  pipeline-cpu.sif

docker run --rm \
  -v "/Users/alice/project/data:/data:ro" \
  -v "/Users/alice/project/results:/results" \
  nolanzz/pipeline:latest

# User-provided data with the GPU image
apptainer run --nv \
  --bind "/Users/alice/project/data:/data" \
  --bind "/Users/alice/project/results:/results" \
  pipeline-gpu.sif

docker run --rm --gpus all \
  -v "/Users/alice/project/data:/data:ro" \
  -v "/Users/alice/project/results:/results" \
  nolanzz/pipeline:gpu</code></pre>

<h2 id="usage">3 Minimal Usage</h2>

<h3>Bundled example dataset</h3>
<p>
  This repository includes the public <strong>ALLAML</strong> dataset as a bundled example for testing the pipeline.
  The original dataset was downloaded from the
  <a href="https://jundongl.github.io/scikit-feature/datasets.html"
     target="_blank"
     rel="noopener noreferrer">Feature Selection Datasets repository</a>.
  The data were split into 80% training data and 20% test data and saved as
  <code>pipeline/data/ALLAML_10.npz</code>.
</p>
<p class="muted">
  The example file contains four arrays:
  <code>x_train</code>, <code>y_train</code>, <code>x_test</code>, and <code>y_test</code>.
  It can therefore be used directly with <code>--preprocess_mode external</code>
  for both feature selection and evaluation examples.
</p>

<p class="muted">
  The pipeline supports two input styles: <strong>auto preprocessing</strong>, where the pipeline processes and splits
  unsplit <code>X</code>/<code>Y</code> data, and <strong>external preprocessing</strong>, where users provide preprocessed
  training arrays and, when evaluation is enabled, test arrays. The bundled ALLAML example uses external preprocessing.
</p>

<h3>3.1 Quick test with the bundled ALLAML example</h3>
<p class="muted">
  Run the commands below from any host directory. These examples use the CPU image by default. The input dataset is
  read from the container image at <code>/app/data/ALLAML_10.npz</code>. The local <code>result/</code> directory is mounted
  to <code>/results</code> so that outputs are saved on the host. For GPU execution, use
  <code>pipeline-gpu.sif</code> together with <code>apptainer run --nv</code>, or use
  <code>nolanzz/pipeline:gpu</code> together with <code>docker run --gpus all</code>.
</p>
<pre class="card-pre"><code>mkdir -p result

apptainer run --bind "$(pwd)/result:/results" pipeline-cpu.sif \
      --input_path /app/data/ALLAML_10.npz \
      --out_dir /results \
      --method EARFS \
      --name ALLAML_10_earfs \
      --preprocess_mode external \
      --task_type binary \
      --is_snp 0 \
      --selected_activate sigmoid \
      --seed 0 \
      --use_evaluation 0 \
      --do_parameter_search 0
    </code></pre>

<h4>Docker equivalent</h4>
<p class="muted">
  The bundled input remains inside the image. Only the local <code>result/</code> directory is mounted, and files
  written to <code>/results</code> inside the container are saved to <code>$PWD/result</code> on the host.
</p>
<pre class="card-pre"><code>mkdir -p result

docker run --rm \
  -v "$PWD/result:/results" \
  nolanzz/pipeline:latest \
  --input_path /app/data/ALLAML_10.npz \
  --out_dir /results \
  --method EARFS \
  --name ALLAML_10_earfs \
  --preprocess_mode external \
  --task_type binary \
  --is_snp 0 \
  --selected_activate sigmoid \
  --seed 0 \
  --use_evaluation 0 \
  --do_parameter_search 0
    </code></pre>

<h3>3.2 Run with your own externally preprocessed data</h3>
<p class="muted">
  Mount the host directory containing your <code>.npz</code> file to a container directory such as <code>/data</code>,
  and mount a separate host directory to <code>/results</code> for persistent outputs. In external mode, the input file
  must contain <code>x_train</code> and <code>y_train</code>; when <code>--use_evaluation 1</code>, it must also contain
  <code>x_test</code> and <code>y_test</code>.
</p>
<pre class="card-pre"><code>mkdir -p results

apptainer run \
  --bind "/absolute/path/to/data:/data" \
  --bind "$(pwd)/results:/results" \
  pipeline-cpu.sif \
  --input_path /data/my_dataset.npz \
  --out_dir /results \
  --method EARFS \
  --name my_dataset \
  --preprocess_mode external \
  --task_type binary \
  --is_snp 0 \
  --selected_activate sigmoid \
  --seed 0 \
  --use_evaluation 0 \
  --do_parameter_search 0
    </code></pre>
<h4>Docker equivalent</h4>
<pre class="card-pre"><code>mkdir -p results

docker run --rm \
  -v "/absolute/path/to/data:/data:ro" \
  -v "$PWD/results:/results" \
  nolanzz/pipeline:latest \
  --input_path /data/my_dataset.npz \
  --out_dir /results \
  --method EARFS \
  --name my_dataset \
  --preprocess_mode external \
  --task_type binary \
  --is_snp 0 \
  --selected_activate sigmoid \
  --seed 0 \
  --use_evaluation 0 \
  --do_parameter_search 0
    </code></pre>
<p class="muted">
  Replace <code>/absolute/path/to/data</code> with the host directory containing your file,
  replace <code>my_dataset.npz</code> with the actual file name, and change <code>--name</code> to a run identifier of your choice.
</p>

<h3>Explanation of key CLI arguments</h3>

<h4>① <code>--input_path</code></h4>
<p>
  Path to the input <code>.npz</code> file inside the container. The required array keys depend on
  <code>--preprocess_mode</code>; see Section 4, <strong>Model options → Data preprocessing → Input file formats</strong>,
  for the complete key requirements.
</p>

<h4>② <code>--preprocess_mode</code></h4>
<p>
  Selects whether preprocessing is handled by the pipeline (<code>auto</code>) or supplied entirely by the user
  (<code>external</code>). See Section 4, <strong>Model options → Data preprocessing</strong>, for the complete behavior
  and parameter conditions.
</p>

<h4>③ <code>--use_evaluation</code></h4>
<ul>
  <li><code>0</code>: feature selection only.</li>
  <li><code>1</code>: run evaluation curves by progressively adding top features and evaluating on an independent test set.</li>
  <li class="muted">See Section 4, Evaluation options, for test-set requirements and method-specific behavior.</li>
</ul>

<h4>④ <code>--out_dir</code> and <code>--name</code></h4>
<p>Control where outputs are written and how this run is identified.</p>

<h4>⑤ <code>--method</code></h4>
<p>
  Select one supported command-line value. Method names are case-sensitive and must be entered exactly as shown below.
  The display names used elsewhere in this document may differ from the corresponding CLI values.
</p>
<table>
  <thead><tr><th>Display name</th><th>CLI value for <code>--method</code></th><th>Example</th></tr></thead>
  <tbody>
    <tr><td>GRACES</td><td><code>GRACES</code></td><td><code>--method GRACES</code></td></tr>
    <tr><td>EAR-FS</td><td><code>EARFS</code></td><td><code>--method EARFS</code></td></tr>
    <tr><td>CancelOut</td><td><code>CANCELOUT</code></td><td><code>--method CANCELOUT</code></td></tr>
    <tr><td>STG</td><td><code>STG</code></td><td><code>--method STG</code></td></tr>
    <tr><td>CAE</td><td><code>CAE</code></td><td><code>--method CAE</code></td></tr>
    <tr><td>DeepLIFT</td><td><code>DeepLIFT</code></td><td><code>--method DeepLIFT</code></td></tr>
    <tr><td>GradientSHAP</td><td><code>GradientShap</code></td><td><code>--method GradientShap</code></td></tr>
    <tr><td>FeatureAblation</td><td><code>FeatureAblation</code></td><td><code>--method FeatureAblation</code></td></tr>
    <tr><td>Occlusion</td><td><code>Occlusion</code></td><td><code>--method Occlusion</code></td></tr>
    <tr><td>LIME</td><td><code>Lime</code></td><td><code>--method Lime</code></td></tr>
  </tbody>
</table>

<h2 id="params">4 All Parameters</h2>

<!--  Required -->
<details open>
  <summary><strong> Input &amp; Required arguments</strong></summary>

  <p class="muted">
    Input array keys depend on <code>--preprocess_mode</code>; see
    <strong>Model options → Data preprocessing → Input file formats</strong> below in Section 4.
  </p>

  <table class="parameter-table">
    <thead>
      <tr>
        <th>Argument</th>
        <th>Type / Default</th>
        <th>Description</th>
        <th>Applicable method / Effective when</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>--input_path</code></td>
        <td><code>str</code>, required</td>
        <td>
          Path to an input <code>.npz</code> file inside the container. The bundled example dataset included in the
          published image is available at <code>/app/data/ALLAML_10.npz</code>. For user-provided data mounted to
          <code>/data</code>, use a path such as <code>/data/my_dataset.npz</code>.
        </td>
        <td>All methods; always required.</td>
      </tr>
      <tr>
        <td><code>--out_dir</code></td>
        <td><code>str</code>, default=<code>result</code></td>
        <td>
          Output root directory inside the container. To preserve results on the host,
          use a mounted path such as <code>/results</code>.
        </td>
        <td>All methods.</td>
      </tr>
      <tr>
        <td><code>--method</code></td>
        <td><code>str</code>, default=<code>EARFS</code></td>
        <td>
          Selects the feature-selection or attribution method. CLI values are case-sensitive;
          see the display-name/CLI-value table in Section 3 for the complete mapping.
        </td>
        <td>All runs; selects the active method.</td>
      </tr>
      <tr>
        <td><code>--name</code></td>
        <td><code>str</code>, default=<code>test</code></td>
        <td>Dataset identifier used to label the run and its outputs.</td>
        <td>All methods.</td>
      </tr>
    </tbody>
  </table>

</details>

<!-- ② Model options -->
<details>
  <summary><strong> Model options</strong></summary>

  <!-- ========================= -->
  <!-- 1. Random seed -->
  <!-- ========================= -->
  <h3>Randomness & reproducibility</h3>
  <table class="parameter-table">
    <thead>
      <tr>
        <th>Argument</th>
        <th>Type / Default</th>
        <th>Description</th>
        <th>Applicable method / Effective when</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>--seed</code></td>
        <td><code>int</code>, default=<code>42</code></td>
        <td>
          Random seed controlling NumPy, PyTorch, and internal data splitting,
          ensuring reproducible feature selection and evaluation results.
        </td>
        <td>All methods.</td>
      </tr>
    </tbody>
  </table>

  <!-- ========================= -->
  <!-- 2. Data preprocessing -->
  <!-- ========================= -->
  <h3>Data preprocessing</h3>
  <table class="parameter-table">
    <thead>
      <tr>
        <th>Argument</th>
        <th>Type / Default</th>
        <th>Description</th>
        <th>Applicable method / Effective when</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>--preprocess_mode</code></td>
        <td><code>str</code>, default=<code>auto</code></td>
        <td>
          Controls how data preprocessing is handled:
          <ul>
            <li><code>auto</code>: the pipeline performs missing-value imputation, train/test splitting, and feature scaling.</li>
            <li><code>external</code>: user provides already preprocessed and split data; the pipeline does not perform imputation, train/test splitting, or feature scaling again.</li>
          </ul>
        </td>
        <td>All methods.</td>
      </tr>

      <tr>
        <td><code>--impute_strategy</code></td>
        <td><code>str</code>, default=<code>none</code></td>
        <td>
          Missing-value imputation strategy for <code>X</code>:
          <code>none</code>, <code>mean</code>, <code>zero</code>, <code>mode</code>.
        </td>
        <td>All methods; only when <code>preprocess_mode=auto</code>.</td>
      </tr>

      <tr>
        <td><code>--scaler</code></td>
        <td><code>str</code>, default=<code>zscore</code></td>
        <td>
          Feature scaling strategy for <code>X</code>:
          <code>none</code>, <code>zscore</code>, or <code>minmax</code>.
          Used only when <code>preprocess_mode=auto</code>. It is not applied in
          <code>preprocess_mode=external</code>.
          For SNP genotype data (<code>is_snp=1</code>), the pipeline preserves the original discrete genotype values
          and does not apply continuous scaling. If <code>zscore</code> or <code>minmax</code> is requested for SNP data,
          the scaling step is skipped and a warning is issued.
        </td>
        <td>All methods; only when <code>preprocess_mode=auto</code> and <code>is_snp=0</code>.</td>
      </tr>

      <tr>
        <td><code>--train_ratio</code></td>
        <td><code>float</code>, default=<code>0.8</code></td>
        <td>
          Proportion of samples assigned to the training set. Used only when
          <code>preprocess_mode=auto</code> and the input provides unsplit <code>X</code> and <code>Y</code>.
          It has no effect when pre-split data are supplied or when <code>preprocess_mode=external</code>.
        </td>
        <td>All methods; only for unsplit <code>X</code>/<code>Y</code> in auto mode.</td>
      </tr>

      <tr>
        <td><code>--drop_constant_features</code></td>
        <td><code>int</code>, default=<code>1</code></td>
        <td>
          Whether to remove features that are constant in the training data:
          <code>1</code>=remove; <code>0</code>=keep. Applied only when
          <code>preprocess_mode=auto</code> and an independent train/test split is created.
        </td>
        <td>All methods; auto mode with a newly created train/test split.</td>
      </tr>

      <tr>
        <td><code>--constant_feature_atol</code></td>
        <td><code>float</code>, default=<code>1e-12</code></td>
        <td>
          Absolute tolerance used to identify features with approximately zero variance in the training data.
        </td>
        <td>All methods; when constant-feature detection is enabled.</td>
      </tr>

      <tr>
        <td><code>--constant_feature_report_name</code></td>
        <td><code>str</code>, default=<code>constant_feature_report.json</code></td>
        <td>
          File name used for the report describing constant-feature detection and removal.
        </td>
        <td>All methods; when constant-feature reporting is produced.</td>
      </tr>
    </tbody>
  </table>

  <h4>Input file formats (controlled by <code>--preprocess_mode</code>)</h4>
  <table>
    <thead>
      <tr>
        <th><code>preprocess_mode</code></th>
        <th>Expected keys in <code>.npz</code></th>
        <th>Behavior</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>auto</code></td>
        <td>
          Preferred: <code>X</code>, <code>Y</code><br>
          Also supported (legacy): <code>x_train</code>, <code>y_train</code>
          (optional <code>x_test</code>, <code>y_test</code>)
        </td>
        <td>
          The preferred unsplit format is <code>X</code>/<code>Y</code>. Legacy pre-split keys are also accepted.
          If no independent test set is available or can be constructed, <code>--use_evaluation</code> is forced to <code>0</code>.
        </td>
      </tr>
      <tr>
        <td><code>external</code></td>
        <td>
          Required: <code>x_train</code>, <code>y_train</code><br>
          If <code>--use_evaluation 1</code>:
          also require <code>x_test</code>, <code>y_test</code>
        </td>
        <td>
          The supplied train/test arrays are used as provided. If <code>--use_evaluation 1</code> is requested without
          <code>x_test</code>/<code>y_test</code>, the pipeline throws an error.
        </td>
      </tr>
    </tbody>
  </table>

  <!-- ========================= -->
  <!-- 3. Task, data type & model -->
  <!-- ========================= -->
  <h3>Task & model configuration</h3>
  <table class="parameter-table">
    <thead>
      <tr>
        <th>Argument</th>
        <th>Type / Default</th>
        <th>Description</th>
        <th>Applicable method / Effective when</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>--task_type</code></td>
        <td><code>str</code>, default=<code>auto</code></td>
        <td>
          Task type: <code>auto</code>, <code>binary</code>, <code>multiclass</code>, <code>regression</code>.
          When set to <code>auto</code>, the pipeline infers the task type and automatically
          fixes label encoding if needed.
        </td>
        <td>All methods.</td>
      </tr>

      <tr>
        <td><code>--is_snp</code></td>
        <td><code>int</code>, default=<code>0</code></td>
        <td>
          Set to <code>1</code> when the input contains discrete SNP genotype values, typically coded as integers.
          The original genotype coding is preserved; see <code>--scaler</code> under Data preprocessing for scaling behavior.
        </td>
        <td>All methods; set to <code>1</code> for SNP genotype data.</td>
      </tr>

      <tr>
        <td><code>--selected_activate</code></td>
        <td><code>str</code>, default=<code>relu</code></td>
        <td>
          Activation function used in the neural network backbone
          (<code>relu</code>, <code>tanh</code>, <code>sigmoid</code>, <code>leakyrelu</code>, <code>elu</code>, <code>gelu</code>).
        </td>
        <td>Neural-network-based methods.</td>
      </tr>

      <tr>
        <td><code>--epochs</code></td>
        <td><code>int</code>, default=<code>100</code></td>
        <td>Maximum number of training epochs.</td>
        <td>Neural-network-based methods.</td>
      </tr>

      <tr>
        <td><code>--batch_size</code></td>
        <td><code>int</code>, default=<code>32</code></td>
        <td>Mini-batch size used during training.</td>
        <td>Neural-network-based methods.</td>
      </tr>

      <tr>
        <td><code>--lr</code></td>
        <td><code>float</code>, default=<code>0.001</code></td>
        <td>Learning rate for the optimizer.</td>
        <td>Neural-network-based methods.</td>
      </tr>

      <tr>
        <td><code>--digits</code></td>
        <td><code>int</code>, default=<code>100</code></td>
        <td>
          Starting divisor used to construct the hidden-layer architecture. For hidden layer <code>i</code>,
          the divisor is approximately <code>digits × shrink_ratio^i</code>, and the layer width is
          <code>max(4, input_dim / divisor)</code>.
        </td>
        <td>Methods using the shared neural-network backbone.</td>
      </tr>

      <tr>
        <td><code>--shrink_ratio</code></td>
        <td><code>float</code>, default=<code>2.0</code></td>
        <td>
          Multiplicative factor used to increase the divisor across successive hidden layers,
          thereby shrinking the layer widths.
        </td>
        <td>Methods using the shared neural-network backbone.</td>
      </tr>

      <tr>
        <td><code>--n_layers</code></td>
        <td><code>int</code>, default=<code>2</code></td>
        <td>Number of hidden layers used when running with fixed hyperparameters.</td>
        <td>Methods using the shared neural-network backbone.</td>
      </tr>

      <tr>
        <td><code>--dropout</code></td>
        <td><code>float</code>, default=<code>0.5</code></td>
        <td>Dropout probability applied to hidden layers.</td>
        <td>Neural-network-based methods.</td>
      </tr>

      <tr>
        <td><code>--weight_decay</code></td>
        <td><code>float</code>, default=<code>0.0</code></td>
        <td>Optimizer weight decay (L2 regularization).</td>
        <td>Neural-network-based methods.</td>
      </tr>

      <tr>
        <td><code>--validation_split</code></td>
        <td><code>float</code>, default=<code>0.2</code></td>
        <td>Fraction of training data used for validation.</td>
        <td>Training workflows that use a validation split.</td>
      </tr>

      <tr>
        <td><code>--patience</code></td>
        <td><code>int</code>, default=<code>10</code></td>
        <td>Number of epochs with no improvement before early stopping.</td>
        <td>Training workflows with early stopping.</td>
      </tr>

      <tr>
        <td><code>--min_delta</code></td>
        <td><code>float</code>, default=<code>0.0</code></td>
        <td>Minimum improvement required to reset early stopping.</td>
        <td>Training workflows with early stopping.</td>
      </tr>

      <tr>
        <td><code>--n_iters</code></td>
        <td><code>int</code>, default=<code>1</code></td>
        <td>
          Number of internal repetitions used by some feature-selection
          and evaluation procedures.
        </td>
        <td>Evaluation and methods that support repeated runs.</td>
      </tr>
    </tbody>
  </table>
</details>

<!-- ③ Evaluation options -->
<details>
  <summary><strong> Evaluation options</strong></summary>

  <p class="muted">
    Evaluation requires an independent test set. Auto mode may disable evaluation if no test split can be created,
    while external mode requires <code>x_test</code> and <code>y_test</code>. CAE uses a method-specific evaluation workflow;
    see <code>--cat_k_select</code> below for details.
  </p>
  <p class="muted">
    To avoid conflicting documentation, defaults for method-specific parameters are stated once in
    <strong>Fixed hyperparameters &amp; method-specific constants</strong>. Repeated references in this section describe only
    their evaluation behavior.
  </p>

  <table class="parameter-table">
    <thead>
      <tr>
        <th>Argument</th>
        <th>Type / Default</th>
        <th>Description</th>
        <th>Applicable method / Effective when</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>--use_evaluation</code></td>
        <td><code>int</code>, default=<code>0</code></td>
        <td><code>0</code>=feature selection only; <code>1</code>=evaluate performance while increasing feature count.</td>
        <td>All methods.</td>
      </tr>
      <tr>
        <td><code>--max_features</code></td>
        <td><code>int</code>, default=<code>100</code></td>
        <td>Upper bound on the number of top-ranked features evaluated in the ordinary evaluation workflow.</td>
        <td>All methods in ordinary evaluation; constrained by <code>max_features_graces</code> for GRACES.</td>
      </tr>
      <tr>
        <td><code>--feature_step</code></td>
        <td><code>int</code>, default=<code>5</code></td>
        <td>Step size used to construct the ordinary sequence of evaluated feature counts.</td>
        <td>All methods in ordinary evaluation; CAE primarily uses <code>cat_k_select</code>.</td>
      </tr>
      <tr>
        <td><code>--max_features_graces</code></td>
        <td><code>int</code>; default listed under <strong>GRACES</strong> fixed parameters</td>
        <td>
          Upper limit on the number of features selected by GRACES during evaluation.
          The required constraint is <code>max_features &lt;= max_features_graces</code>.
        </td>
        <td>GRACES only.</td>
      </tr>
      <tr>
        <td><code>--cat_k_select</code></td>
        <td><code>str</code>; default listed under <strong>CAE</strong> fixed parameters</td>
        <td>
          Comma-separated feature counts used during CAE evaluation. CAE runs separately for each K value,
          and this list is the primary control for the evaluated feature counts.
        </td>
        <td>CAE only.</td>
      </tr>
    </tbody>
  </table>

</details>

<!-- ④ Optuna hyperparameter options -->
<details>
  <summary><strong> Optuna hyperparameter options</strong></summary>

  <p class="muted">
    If running on SLURM, <code>--n_jobs</code> is auto-adjusted when <code>SLURM_CPUS_PER_TASK</code> is available:
    <code>n_jobs = max(1, SLURM_CPUS_PER_TASK - 2)</code>.
  </p>

  <table class="parameter-table">
    <thead>
      <tr>
        <th>Argument</th>
        <th>Type / Default</th>
        <th>Description</th>
        <th>Applicable method / Effective when</th>
      </tr>
    </thead>
    <tbody>
      <tr><td><code>--do_parameter_search</code></td><td><code>int</code>, default=<code>0</code></td><td>Use Optuna: <code>0</code>=no; <code>1</code>=search.</td><td>All methods that support Optuna search.</td></tr>
      <tr><td><code>--eval_metric</code></td><td><code>str</code>, default=<code>loss</code></td><td>Metric for search: <code>acc</code>/<code>r2</code>/<code>mse</code>/<code>mae</code>/<code>loss</code>.</td><td>Only when <code>do_parameter_search=1</code>.</td></tr>
      <tr><td><code>--n_trials</code></td><td><code>int</code>, default=<code>20</code></td><td>Number of trials.</td><td>Only when <code>do_parameter_search=1</code>.</td></tr>
      <tr><td><code>--n_jobs</code></td><td><code>int</code>, default=<code>1</code></td><td>Parallel jobs for Optuna (may be auto-set on SLURM).</td><td>Only when <code>do_parameter_search=1</code>.</td></tr>
      <tr>
        <td><code>--use_cv</code></td>
        <td><code>int</code>, default=<code>1</code></td>
        <td>
          Whether Optuna evaluates each trial using cross-validation:
          <code>1</code>=cross-validation; <code>0</code>=a single validation split.
        </td>
        <td>Only when <code>do_parameter_search=1</code>.</td>
      </tr>
      <tr>
        <td><code>--n_splits</code></td>
        <td><code>int</code>, default=<code>5</code></td>
        <td>Number of folds used for cross-validation during Optuna search.</td>
        <td>Only when <code>do_parameter_search=1</code> and <code>use_cv=1</code>.</td>
      </tr>

      <tr><td><code>--batch_size_list</code></td><td><code>str</code>, default=<code>8,16,32</code></td><td>Candidates for batch size (comma-separated).</td><td>Optuna search for neural-network-based methods.</td></tr>

      <tr>
        <td><code>--digits_list</code></td>
        <td><code>str</code>, default=<code>100,200,500,1000</code></td>
        <td>
          Comma-separated candidate starting divisors used to construct multi-layer hidden architectures together with
          <code>--shrink_ratio_list</code>, <code>--min_layers</code>, and <code>--max_layers</code>.
        </td>
        <td>Optuna search for methods using the shared neural-network backbone.</td>
      </tr>

      <tr>
        <td><code>--shrink_ratio_list</code></td>
        <td><code>str</code>, default=<code>1</code></td>
        <td>Comma-separated candidate shrink ratios for constructing multi-layer network architectures.</td>
        <td>Optuna search for methods using the shared neural-network backbone.</td>
      </tr>
      <tr>
        <td><code>--min_layers</code></td>
        <td><code>int</code>, default=<code>2</code></td>
        <td>Minimum number of hidden layers considered during hyperparameter search.</td>
        <td>Optuna search for methods using the shared neural-network backbone.</td>
      </tr>
      <tr>
        <td><code>--max_layers</code></td>
        <td><code>int</code>, default=<code>2</code></td>
        <td>Maximum number of hidden layers considered during hyperparameter search.</td>
        <td>Optuna search for methods using the shared neural-network backbone.</td>
      </tr>

      <tr><td><code>--lr_min</code> / <code>--lr_max</code></td><td><code>float</code>, default=<code>1e-5</code> / <code>1e-1</code></td><td>Learning-rate search range.</td><td>Optuna search for neural-network-based methods.</td></tr>
      <tr><td><code>--weight_decay_min</code> / <code>--weight_decay_max</code></td><td><code>float</code>, default=<code>0.0</code> / <code>1e-1</code></td><td>Weight-decay search range.</td><td>Optuna search for neural-network-based methods.</td></tr>
      <tr><td><code>--dropout_min</code> / <code>--dropout_max</code></td><td><code>float</code>, default=<code>0.2</code> / <code>0.8</code></td><td>Dropout search range.</td><td>Optuna search for neural-network-based methods.</td></tr>
    </tbody>
  </table>

<h3 class="muted">CancelOut-specific search</h3>
      <table class="parameter-table">
        <thead><tr><th>Argument</th><th>Type / Default</th><th>Description</th><th>Applicable method / Effective when</th></tr></thead>
        <tbody>
          <tr><td><code>--lambda_1_min</code> / <code>--lambda_1_max</code></td><td><code>float</code>, default=<code>1e-5</code> / <code>1e-1</code></td><td>L1 regularization search range.</td><td>CancelOut only; when <code>do_parameter_search=1</code>.</td></tr>
          <tr><td><code>--lambda_2_min</code> / <code>--lambda_2_max</code></td><td><code>float</code>, default=<code>1e-5</code> / <code>1e-1</code></td><td>Variance regularization search range.</td><td>CancelOut only; when <code>do_parameter_search=1</code>.</td></tr>
          <tr><td><code>--cancelout_init</code></td><td><code>float</code> or <code>None</code>, default=<code>None</code></td><td>Initial CancelOut weights (<code>None</code> uses default initialization logic).</td><td>CancelOut only; when <code>do_parameter_search=1</code>.</td></tr>
          <tr><td><code>--search_cancelout_init</code></td><td><code>int</code>, default=<code>0</code></td><td>Search the best initialization: <code>1</code>=yes.</td><td>CancelOut only; when <code>do_parameter_search=1</code>.</td></tr>
        </tbody>
      </table>

      <h3 class="muted">STG-specific search</h3>
      <table class="parameter-table">
        <thead><tr><th>Argument</th><th>Type / Default</th><th>Description</th><th>Applicable method / Effective when</th></tr></thead>
        <tbody>
          <tr>
            <td><code>--stg_sigma_min</code> / <code>--stg_sigma_max</code></td>
            <td><code>float</code>, default=<code>0.1</code> / <code>1.0</code></td>
            <td>Search range for the Gaussian-noise standard deviation used in the stochastic gates.</td>
            <td>STG only; when <code>do_parameter_search=1</code>.</td>
          </tr>
          <tr>
            <td><code>--stg_lam_min</code> / <code>--stg_lam_max</code></td>
            <td><code>float</code>, default=<code>1e-4</code> / <code>1.0</code></td>
            <td>Search range for the STG sparsity-regularization strength.</td>
            <td>STG only; when <code>do_parameter_search=1</code>.</td>
          </tr>
        </tbody>
      </table>

      <h3 class="muted">CAE-specific search</h3>
      <table class="parameter-table">
        <thead><tr><th>Argument</th><th>Type / Default</th><th>Description</th><th>Applicable method / Effective when</th></tr></thead>
        <tbody>
          <tr>
            <td><code>--cat_start_temp_min</code> / <code>--cat_start_temp_max</code></td>
            <td><code>float</code>, default=<code>5.0</code> / <code>20.0</code></td>
            <td>Search range for the initial Concrete-selector temperature.</td>
            <td>CAE only; when <code>do_parameter_search=1</code>.</td>
          </tr>
          <tr>
            <td><code>--cat_min_temp_min</code> / <code>--cat_min_temp_max</code></td>
            <td><code>float</code>, default=<code>0.01</code> / <code>1.0</code></td>
            <td>Search range for the minimum temperature reached during selector annealing.</td>
            <td>CAE only; when <code>do_parameter_search=1</code>.</td>
          </tr>
        </tbody>
      </table>

      <h3 class="muted">EAR-FS-specific search</h3>
      <table class="parameter-table">
        <thead><tr><th>Argument</th><th>Type / Default</th><th>Description</th><th>Applicable method / Effective when</th></tr></thead>
        <tbody>
          <tr><td><code>--lambda_fs_min</code> / <code>--lambda_fs_max</code></td><td><code>float</code>, default=<code>1e-5</code> / <code>1e-1</code></td><td>EARFS feature-regularization search range.</td><td>EAR-FS only; when <code>do_parameter_search=1</code>.</td></tr>
        </tbody>
      </table>

      <h3 class="muted">GRACES-specific search</h3>
      <table class="parameter-table">
        <thead><tr><th>Argument</th><th>Type / Default</th><th>Description</th><th>Applicable method / Effective when</th></tr></thead>
        <tbody>
          <tr><td><code>--alpha_min</code> / <code>--alpha_max</code></td><td><code>float</code>, default=<code>0.8</code> / <code>0.99</code></td><td>Cosine-similarity threshold range for building the graph (<code>cos &gt; alpha → edge</code>).</td><td>GRACES only; when <code>do_parameter_search=1</code>.</td></tr>
          <tr><td><code>--f_correct_list</code></td><td><code>str</code>, default=<code>0,0.1,0.5,0.9</code></td><td>Candidate balancing coefficients between graph-derived score and F-test score.</td><td>GRACES only; when <code>do_parameter_search=1</code>.</td></tr>
        </tbody>
      </table>
    </details>

    <!-- ⑤ Fixed hyperparameters & method-specific constants -->
    <details>
      <summary><strong> Fixed hyperparameters &amp; method-specific constants</strong></summary>
      <h3 class="muted">EAR-FS</h3>
      <table class="parameter-table">
        <thead><tr><th>Argument</th><th>Type / Default</th><th>Description</th><th>Applicable method / Effective when</th></tr></thead>
        <tbody>
          <tr><td><code>--lambda_fs</code></td><td><code>float</code>, default=<code>1e-2</code></td><td>Feature-regularization strength.</td><td>EAR-FS only.</td></tr>
        </tbody>
      </table>

      <h3 class="muted">CancelOut</h3>
      <table class="parameter-table">
        <thead><tr><th>Argument</th><th>Type / Default</th><th>Description</th><th>Applicable method / Effective when</th></tr></thead>
        <tbody>
          <tr><td><code>--lambda_1</code></td><td><code>float</code>, default=<code>0.001</code></td><td>L1 regularization.</td><td>CancelOut only.</td></tr>
          <tr><td><code>--lambda_2</code></td><td><code>float</code>, default=<code>0.001</code></td><td>Variance regularization.</td><td>CancelOut only.</td></tr>
        </tbody>
      </table>

      <h3 class="muted">STG</h3>
      <p>
        STG (Stochastic Gates) performs embedded feature selection by learning a stochastic gate for each input feature.
        Features with larger learned gate values are ranked as more important.
      </p>
      <table class="parameter-table">
        <thead><tr><th>Argument</th><th>Type / Default</th><th>Description</th><th>Applicable method / Effective when</th></tr></thead>
        <tbody>
          <tr>
            <td><code>--stg_sigma</code></td>
            <td><code>float</code>, default=<code>0.5</code></td>
            <td>Standard deviation of the Gaussian noise used in the stochastic gates.</td>
            <td>STG only.</td>
          </tr>
          <tr>
            <td><code>--stg_lam</code></td>
            <td><code>float</code>, default=<code>0.1</code></td>
            <td>Regularization strength controlling sparsity of the learned stochastic gates.</td>
            <td>STG only.</td>
          </tr>
        </tbody>
      </table>

      <h3 class="muted">CAE</h3>
      <p>
        CAE (Concrete Autoencoder) performs embedded feature selection by learning a differentiable selector based on
        Concrete-distribution gates. During training, the selector gradually sharpens as the temperature decreases,
        allowing the model to identify a fixed number of informative input features. The selected feature count is
        controlled by <code>--cat_k_select</code>.
      </p>
      <table class="parameter-table">
        <thead><tr><th>Argument</th><th>Type / Default</th><th>Description</th><th>Applicable method / Effective when</th></tr></thead>
        <tbody>
          <tr>
            <td><code>--cat_k_select</code></td>
            <td><code>str</code>, default=<code>26,130,260,390,520,650,779,909,1039</code></td>
            <td>Comma-separated feature counts used by CAE. See Evaluation options for evaluation-specific behavior.</td>
            <td>CAE only.</td>
          </tr>
          <tr>
            <td><code>--cat_start_temp</code></td>
            <td><code>float</code>, default=<code>10.0</code></td>
            <td>Initial temperature of the Concrete selector.</td>
            <td>CAE only.</td>
          </tr>
          <tr>
            <td><code>--cat_min_temp</code></td>
            <td><code>float</code>, default=<code>0.1</code></td>
            <td>Minimum temperature reached during selector annealing.</td>
            <td>CAE only.</td>
          </tr>
          <tr>
            <td><code>--cat_tryout_limit</code></td>
            <td><code>int</code>, default=<code>1</code></td>
            <td>Maximum number of CAE tryouts for each K value.</td>
            <td>CAE only.</td>
          </tr>
          <tr>
            <td><code>--cat_selector_mode</code></td>
            <td><code>str</code>, default=<code>supervised</code></td>
            <td>Selector mode: <code>unsupervised</code> or <code>supervised</code>.</td>
            <td>CAE only.</td>
          </tr>
        </tbody>
      </table>

      <h3 class="muted">GRACES</h3>
      <table class="parameter-table">
        <thead><tr><th>Argument</th><th>Type / Default</th><th>Description</th><th>Applicable method / Effective when</th></tr></thead>
        <tbody>
          <tr>
            <td><code>--max_features_graces</code></td>
            <td><code>int</code>, default=<code>200</code></td>
            <td>
              Maximum number of features selected by GRACES. See Evaluation options for the constraint applied during evaluation.
            </td>
            <td>GRACES only.</td>
          </tr>
          <tr>
            <td><code>--graph_layer_mode</code></td>
            <td><code>str</code>, default=<code>one_sage_then_mlp</code></td>
            <td>
              Controls the hidden-layer implementation in GRACES. Available values:
              <code>all_sage</code> uses SAGEConv throughout the hidden layers;
              <code>one_sage_then_mlp</code> uses one SAGEConv transition followed by standard MLP layers.
            </td>
            <td>GRACES only.</td>
          </tr>
          <tr><td><code>--f_correct</code></td><td><code>float</code>, default=<code>0.5</code></td><td>Balance between graph-derived score and F-test score.</td><td>GRACES only.</td></tr>
          <tr><td><code>--alpha</code></td><td><code>float</code>, default=<code>0.95</code></td><td>Cosine-similarity threshold when building the graph.</td><td>GRACES only.</td></tr>
          <tr><td><code>--sigma</code></td><td><code>float</code>, default=<code>0.0</code></td><td>Variance of Gaussian noise matrix used in scoring (0.0 disables).</td><td>GRACES only.</td></tr>
          <tr><td><code>--n_dropouts</code></td><td><code>int</code>, default=<code>10</code></td><td>Number of stochastic dropout passes for gradient-based scoring.</td><td>GRACES only.</td></tr>
          <tr><td><code>--q</code></td><td><code>int</code>, default=<code>2</code></td><td>q-norm for gradient aggregation (1=L1, 2=L2, large q≈L∞).</td><td>GRACES only.</td></tr>
        </tbody>
      </table>

    </details>


    <!-- ⑤ Results -->
<h2 id="results">5 Results (Outputs)</h2>

<p>
  The pipeline produces two output modes: <b>feature selection only</b> and <b>feature selection + evaluation</b>.
  Each mode has two variants depending on whether Optuna hyperparameter search is enabled (<code>do_parameter_search</code>).
  Below we document the output folder structure and the meaning of each file.
</p>



<h3 id="results-earfs-fs-fixed">5.1 Feature selection only (fixed params; <code>do_parameter_search=0</code>) (<code>EAR-FS</code>)</h3>

<pre class="card-pre"><code>&lt;out_dir&gt;/
└── &lt;name&gt;/
    └── result&lt;activation&gt;/
        └── result_&lt;seed&gt;/
            └── EARFS/
                ├── EARFS_&lt;name&gt;_weights.npy
                └── EARFS_&lt;name&gt;_idx.npy
</code></pre>

<ul>
  <li><code>EARFS_&lt;name&gt;_weights.npy</code>:
    Feature selection weights for all features (selection rates).
    These values come from <code>sigmoid(selection_rate)</code> in the feature selection layer.
  </li>
  <li><code>EARFS_&lt;name&gt;_idx.npy</code>:
    Feature ranking as integer indices sorted by decreasing weight.
    The first index is the most important feature selected by EAR-FS.
  </li>
</ul>

<details>
  <summary><b>How to use these outputs</b></summary>
  <ul>
    <li>To get top-K selected features: load <code>EARFS_&lt;name&gt;_idx.npy</code> and take <code>idx[:K]</code>.</li>
    <li>To inspect selection strength: load <code>EARFS_&lt;name&gt;_weights.npy</code> and compare weights across features.</li>
  </ul>
</details>

<hr/>

<h3 id="results-earfs-fs-optuna">5.2 Feature selection only (Optuna search; <code>do_parameter_search=1</code>) (<code>EAR-FS</code>)</h3>

<pre class="card-pre"><code>&lt;out_dir&gt;/
└── &lt;name&gt;/
    └── result&lt;activation&gt;/
        └── result_&lt;seed&gt;/
            └── EARFS/
                ├── EARFS_&lt;name&gt;_weights.npy
                ├── EARFS_&lt;name&gt;_idx.npy
                └── optuna_search/
                    ├── optuna_EARFS.db
                    ├── best_params.txt
                    └── optuna_study.pkl
</code></pre>

<ul>
  <li><code>EARFS_&lt;name&gt;_weights.npy</code>:
    Final feature selection weights from the model retrained using the best Optuna hyperparameters.
  </li>
  <li><code>EARFS_&lt;name&gt;_idx.npy</code>:
    Final feature ranking (indices sorted by decreasing weights).
  </li>
  <li><code>optuna_search/</code>:
    Folder containing Optuna search artifacts.
    <ul>
      <li><code>optuna_EARFS.db</code>:
        SQLite database created by Optuna (supports persistence and parallel workers).
      </li>
      <li><code>best_params.txt</code>:
        Best hyperparameters found by Optuna plus timing summary.
      </li>
      <li><code>optuna_study.pkl</code>:
        Serialized Optuna study object (useful for post-hoc analysis and plotting).
      </li>
    </ul>
  </li>
</ul>

<details>
  <summary><b>Typical Optuna search space (EAR-FS)</b></summary>
  <ul>
    <li><code>lambda_fs</code>, <code>weight_decay</code>, <code>dropout</code>, <code>lr</code>, <code>batch_size</code>, <code>digits</code></li>
    <li><code>digits</code> is the starting divisor used to construct the multi-layer hidden architecture together with <code>shrink_ratio</code> and <code>n_layers</code>.</li>
  </ul>
</details>

<hr/>

<h3 id="results-earfs-eval-fixed">5.3 Feature selection + evaluation (fixed params; <code>do_parameter_search=0</code>) (<code>EAR-FS</code>)</h3>

<pre class="card-pre"><code>&lt;out_dir&gt;/
└── &lt;name&gt;/
    └── result&lt;activation&gt;/
        └── result_&lt;seed&gt;/
            └── EARFS/
                ├── EARFS_&lt;name&gt;_results.npz
                └── evaluation/
                    ├── checkpoints/
                    │   └── checkpoint.npz
                    ├── iterations/
                    │   ├── feature_ranking_iter_0.npz
                    │   ├── full_results_iter_0.npz
                    │   ├── feature_ranking_iter_1.npz
                    │   ├── full_results_iter_1.npz
                    │   └── ...
                    └── feature_evaluations/
                        ├── features_&lt;K&gt;_iter_0.npz
                        ├── features_&lt;K&gt;_iter_1.npz
                        └── ...
</code></pre>

<ul>
  <li><code>EARFS_&lt;name&gt;_results.npz</code>:
    Aggregated evaluation results across all iterations and evaluated feature counts
    (includes raw matrices + mean/std summaries).
  </li>
  <li><code>evaluation/checkpoints/checkpoint.npz</code>:
    Resume checkpoint containing partial results and progress status
    (useful if jobs are interrupted).
  </li>
  <li><code>evaluation/iterations/feature_ranking_iter_i.npz</code>:
    Feature ranking and weights produced in iteration <code>i</code>, including the seed.
  </li>
  <li><code>evaluation/iterations/full_results_iter_i.npz</code>:
    All scores and losses for iteration <code>i</code> across the entire <code>feature_sequence</code>.
  </li>
  <li><code>evaluation/feature_evaluations/features_&lt;K&gt;_iter_i.npz</code>:
    Detailed evaluation for “top-K features” at iteration <code>i</code>, including selected indices,
    test score, CV loss, and the best dropout/L2 used during evaluation.
  </li>
</ul>

<details>
  <summary><b>What is inside <code>EARFS_&lt;name&gt;_results.npz</code>?</b></summary>
  <ul>
    <li><code>test_results</code>:
      2D array <code>(n_steps, n_iters)</code>, test score for each evaluated K and each iteration.
      (Score name depends on task: AUC / AUC_macro / PearsonR.)
    </li>
    <li><code>cv_val_losses</code>:
      2D array <code>(n_steps, n_iters)</code>, best CV validation loss for each evaluated K and iteration.
    </li>
    <li><code>weights</code>:
      Object array of length <code>n_iters</code>, each element is the full weight vector for that iteration.
    </li>
    <li><code>indices</code>:
      Object array of length <code>n_iters</code>, each element is the full feature ranking for that iteration.
    </li>
    <li><code>best_params</code>:
      <code>None</code> for fixed-params mode (present only when Optuna is used).
    </li>
    <li><code>mean_test_results</code>, <code>std_test_results</code>:
      Mean and standard deviation across iterations for each evaluated K.
    </li>
    <li><code>mean_cv_val_losses</code>, <code>std_cv_val_losses</code>:
      Mean and standard deviation across iterations for CV validation loss at each evaluated K.
    </li>
    <li><code>feature_sequence</code>:
      The evaluated feature counts (K values) determined by <code>--max_features</code> and <code>--feature_step</code>.
    </li>
    <li><code>total_features</code>, <code>max_features</code>, <code>n_steps</code>, <code>seeds</code>:
      Metadata about the evaluation configuration.
    </li>
    <li><code>config</code>:
      Dictionary-like object containing <code>n_folds</code>, <code>n_iters</code>, <code>dropout_probs</code>, <code>l2_lambdas</code>.
    </li>
    <li><code>timestamp</code>:
      A timestamp string (e.g. <code>YYYYMMDD_HHMMSS</code>) marking when the result file was produced.
    </li>
    <li><code>per_step_report_metrics</code>:
      Object array storing per-K, per-iteration detailed metrics returned by <code>evaluate_feature_set</code>.
    </li>
  </ul>
</details>

<details>
  <summary><b>What is inside per-iteration files?</b></summary>
  <ul>
    <li><code>feature_ranking_iter_i.npz</code> contains:
      <code>feature_indices</code>, <code>weights</code>, <code>iter</code>, <code>seed</code>.
    </li>
    <li><code>full_results_iter_i.npz</code> contains:
      <code>iter</code>, <code>test_results</code> (1D over K),
      <code>cv_val_losses</code> (1D over K),
      <code>feature_indices</code>, <code>feature_weights</code>,
      <code>feature_sequence</code>, <code>seed</code>, <code>timestamp</code>,
      <code>report_metrics_column</code> (object array over K).
    </li>
    <li><code>features_&lt;K&gt;_iter_i.npz</code> contains:
      <code>num_features</code>, <code>iter</code>, <code>selected_features</code>,
      <code>test_score</code>, <code>best_cv_val_loss</code>, <code>best_dropout</code>, <code>best_l2</code>,
      <code>report_metrics</code>.
    </li>
  </ul>
</details>

<hr/>

<h3 id="results-earfs-eval-optuna">5.4 Feature selection + evaluation (Optuna search; <code>do_parameter_search=1</code>) (<code>EAR-FS</code>)</h3>

<pre class="card-pre"><code>&lt;out_dir&gt;/
└── &lt;name&gt;/
    └── result&lt;activation&gt;/
        └── result_&lt;seed&gt;/
            └── EARFS/
                ├── EARFS_&lt;name&gt;_results.npz
                └── evaluation/
                    ├── checkpoints/
                    │   └── checkpoint.npz
                    ├── iterations/
                    │   ├── feature_ranking_iter_0.npz
                    │   ├── full_results_iter_0.npz
                    │   └── ...
                    ├── feature_evaluations/
                    │   ├── features_&lt;K&gt;_iter_0.npz
                    │   └── ...
                    └── optuna_iter_0/
                        └── optuna_search/
                            ├── optuna_EARFS.db
                            ├── best_params.txt
                            └── optuna_study.pkl
</code></pre>

<ul>
  <li><code>EARFS_&lt;name&gt;_results.npz</code>:
    Same aggregated result container as in 5.3, but now includes Optuna-driven behavior
    (the file may contain <code>best_params</code> and evaluation reflects the parameters selected per iteration).
  </li>
  <li><code>evaluation/checkpoints/</code>, <code>evaluation/iterations/</code>, <code>evaluation/feature_evaluations/</code>:
    Same meaning as 5.3.
  </li>
  <li><code>evaluation/optuna_iter_i/optuna_search/</code>:
    Optuna artifacts for iteration <code>i</code> (each iteration can run its own hyperparameter search before producing that iteration’s ranking).
    <ul>
      <li><code>optuna_EARFS.db</code>: SQLite DB for that iteration’s Optuna run.</li>
      <li><code>best_params.txt</code>: best hyperparameters found in that iteration.</li>
      <li><code>optuna_study.pkl</code>: serialized Optuna study for that iteration.</li>
    </ul>
  </li>
</ul>

<details>
  <summary><b>Notes</b></summary>
  <ul>
    <li>
      In evaluation mode, feature ranking is produced first (EAR-FS training),
      then evaluation runs for multiple K values using <code>evaluate_feature_set(...)</code>.
    </li>
    <li>
      The reported test score name depends on <code>task_type</code>:
      <code>binary</code> → AUC, <code>multiclass</code> → macro-AUC, <code>regression</code> → Pearson correlation.
    </li>
  </ul>
</details>

<hr/>

<h3 id="results-cae">5.5 CAE outputs</h3>

<p>
  CAE runs separately for each feature-count value specified by <code>--cat_k_select</code>.
  Each K value is stored in its own subdirectory under the CAE method directory:
</p>

<pre class="card-pre"><code>&lt;out_dir&gt;/
└── &lt;name&gt;/
    └── result&lt;activation&gt;/
        └── result_&lt;seed&gt;/
            └── CAE/
                ├── &lt;K_1&gt;/
                ├── &lt;K_2&gt;/
                ├── &lt;K_3&gt;/
                └── ...
</code></pre>

<p class="muted">
  For a quick test with the bundled ALLAML dataset, use a smaller setting such as
  <code>--cat_k_select 5,10,20</code>. This creates the subdirectories
  <code>CAE/5/</code>, <code>CAE/10/</code>, and <code>CAE/20/</code>.
  For a full analysis, choose K values according to the total number of input features and the desired selection depths.
</p>

<h4>Feature selection only (fixed parameters)</h4>

<pre class="card-pre"><code>&lt;K&gt;/
├── CAE_&lt;name&gt;_results.npy
└── CAE_&lt;name&gt;_weights.npy
</code></pre>

<ul>
  <li><code>CAE_&lt;name&gt;_results.npy</code>: hard-selected feature indices for the requested K value.</li>
  <li><code>CAE_&lt;name&gt;_weights.npy</code>: feature scores produced by the trained CAE selector.</li>
</ul>

<h4>Feature selection only (Optuna search)</h4>

<pre class="card-pre"><code>&lt;K&gt;/
├── CAE_&lt;name&gt;_results.npy
├── CAE_&lt;name&gt;_weights.npy
└── optuna_search/
    ├── optuna_CAE.db
    ├── best_params.txt
    └── optuna_study.pkl
</code></pre>

<ul>
  <li><code>optuna_CAE.db</code>: SQLite database containing the Optuna study and trial history.</li>
  <li><code>best_params.txt</code>: best hyperparameters and the corresponding optimization summary.</li>
  <li><code>optuna_study.pkl</code>: serialized Optuna study object for later inspection and plotting.</li>
</ul>

<h4>Feature selection + evaluation (fixed parameters)</h4>

<pre class="card-pre"><code>&lt;K&gt;/
├── CAE_&lt;name&gt;_results.npz
└── evaluation/
    ├── checkpoints/
    │   └── checkpoint.npz
    ├── iterations/
    │   ├── feature_ranking_iter_0.npz
    │   ├── full_results_iter_0.npz
    │   └── ...
    └── feature_evaluations/
        ├── features_&lt;K&gt;_iter_0.npz
        └── ...
</code></pre>

<h4>Feature selection + Optuna + evaluation</h4>

<pre class="card-pre"><code>&lt;K&gt;/
├── CAE_&lt;name&gt;_results.npz
└── evaluation/
    ├── checkpoints/
    │   └── checkpoint.npz
    ├── iterations/
    │   ├── feature_ranking_iter_0.npz
    │   ├── full_results_iter_0.npz
    │   └── ...
    ├── feature_evaluations/
    │   ├── features_&lt;K&gt;_iter_0.npz
    │   └── ...
    └── optuna_iter_0/
        └── optuna_search/
            ├── optuna_CAE.db
            ├── best_params.txt
            └── optuna_study.pkl
</code></pre>

<p class="muted">
  When <code>--n_iters</code> is greater than 1, CAE creates one
  <code>evaluation/optuna_iter_&lt;i&gt;/optuna_search/</code> directory for each evaluation iteration.
  Duplicate CAE <code>.npy</code> files, timing files, auxiliary JSON files, and empty trial/fold directories are not produced in evaluation mode.
</p>

    <!-- 6 Advanced usage -->
    <h2 id="adv">6 Advanced Usage</h2>
    <p>Below are four common combinations, followed by a method-specific CAE quick test:</p>
    <p class="muted">
      The values <code>--max_features 20</code>, <code>--feature_step 5</code>, and
      <code>--max_features_graces 20</code> used in the evaluation examples are lightweight settings intended
      for a quick test with the bundled dataset. For a full analysis, increase these values according to the
      total number of input features and the desired evaluation resolution.
    </p>
    <ul>
      <li><code>--use_evaluation</code>: whether to enable <code>evaluation</code>.</li>
      <li><code>--do_parameter_search</code>: whether to run <code>hyperparameter search</code>.</li>
    </ul>

    <!-- 6.1 no eval, no hyper -->
    <h3>6.1 No evaluation, no hyperparameters</h3>
    <p>The simplest and fastest way to generate a feature ranking.</p>
    <pre class="card-pre"><code>mkdir -p result

    apptainer run --bind "$(pwd)/result:/results" pipeline-cpu.sif \
      --input_path /app/data/ALLAML_10.npz \
      --out_dir /results \
      --method EARFS \
      --name ALLAML_10_earfs_fixed \
      --preprocess_mode external \
      --task_type binary \
      --is_snp 0 \
      --selected_activate sigmoid \
      --seed 0 \
      --use_evaluation 0 \
      --do_parameter_search 0 \
      --epochs 100 \
      --batch_size 32 \
      --lr 0.001 \
      --dropout 0.5 \
      --weight_decay 0.0
</code></pre>
    <h4>Docker equivalent</h4>
    <pre class="card-pre"><code>mkdir -p result

docker run --rm \
  -v "$PWD/result:/results" \
  nolanzz/pipeline:latest \
  --input_path /app/data/ALLAML_10.npz \
  --out_dir /results \
  --method EARFS \
  --name ALLAML_10_earfs_fixed \
  --preprocess_mode external \
  --task_type binary \
  --is_snp 0 \
  --selected_activate sigmoid \
  --seed 0 \
  --use_evaluation 0 \
  --do_parameter_search 0 \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.001 \
  --dropout 0.5 \
  --weight_decay 0.0
</code></pre>
    <ul>
      <li><strong>Switch to CancelOut:</strong> focus on <code>--lambda_1</code>, <code>--lambda_2</code>.</li>
      <li><strong>Switch to GRACES:</strong> focus on <code>--alpha</code>, <code>--f_correct</code>, <code>--q</code>, <code>--sigma</code>, <code>--n_dropouts</code>, <code>--max_features_graces</code>.</li>
    </ul>

    <!-- 6.2 no eval, hyper -->
    <h3>6.2 No evaluation, with hyperparameters (Optuna)</h3>
    <p>Performs feature selection only, but first searches hyperparameters via Optuna.</p>
    <pre class="card-pre"><code>mkdir -p result

    apptainer run --bind "$(pwd)/result:/results" pipeline-cpu.sif \
      --input_path /app/data/ALLAML_10.npz \
      --out_dir /results \
      --method EARFS \
      --name ALLAML_10_earfs_optuna \
      --preprocess_mode external \
      --task_type binary \
      --is_snp 0 \
      --selected_activate sigmoid \
      --seed 0 \
      --use_evaluation 0 \
      --do_parameter_search 1 \
      --n_trials 30 \
      --n_jobs 4 \
      --use_cv 1 \
      --n_splits 5 \
      --eval_metric loss \
      --lr_min 1e-5 \
      --lr_max 1e-1 \
      --dropout_min 0.2 \
      --dropout_max 0.8 \
      --weight_decay_min 0.0 \
      --weight_decay_max 1e-1 \
      --batch_size_list 8,16,32 \
      --digits_list 100,200,500,1000
</code></pre>
    <h4>Docker equivalent</h4>
    <pre class="card-pre"><code>mkdir -p result

docker run --rm \
  -v "$PWD/result:/results" \
  nolanzz/pipeline:latest \
  --input_path /app/data/ALLAML_10.npz \
  --out_dir /results \
  --method EARFS \
  --name ALLAML_10_earfs_optuna \
  --preprocess_mode external \
  --task_type binary \
  --is_snp 0 \
  --selected_activate sigmoid \
  --seed 0 \
  --use_evaluation 0 \
  --do_parameter_search 1 \
  --n_trials 30 \
  --n_jobs 4 \
  --use_cv 1 \
  --n_splits 5 \
  --eval_metric loss \
  --lr_min 1e-5 \
  --lr_max 1e-1 \
  --dropout_min 0.2 \
  --dropout_max 0.8 \
  --weight_decay_min 0.0 \
  --weight_decay_max 1e-1 \
  --batch_size_list 8,16,32 \
  --digits_list 100,200,500,1000
</code></pre>
    <ul>
      <li><strong>CancelOut-specific:</strong> add <code>--lambda_1_min/max</code>, <code>--lambda_2_min/max</code>, <code>--search_cancelout_init</code>, <code>--cancelout_init</code>.</li>
      <li><strong>GRACES-specific:</strong> include <code>--alpha_min/max</code> and <code>--f_correct_list</code> (override fixed <code>--alpha</code>, <code>--f_correct</code>).</li>
    </ul>
    <pre class="card-pre"><code># Example: CancelOut + Optuna
mkdir -p result

    apptainer run --bind "$(pwd)/result:/results" pipeline-cpu.sif \
  --input_path /app/data/ALLAML_10.npz \
  --out_dir /results \
  --method CANCELOUT \
  --name ALLAML_10_cancelout_optuna \
  --preprocess_mode external \
  --task_type binary \
  --is_snp 0 \
  --selected_activate sigmoid \
  --seed 0 \
  --use_evaluation 0 \
  --do_parameter_search 1 \
  --n_trials 40 \
  --n_jobs 4 \
  --use_cv 1 \
  --n_splits 5 \
  --lr_min 1e-5 \
  --lr_max 1e-2 \
  --dropout_min 0.2 \
  --dropout_max 0.7 \
  --weight_decay_min 0.0 \
  --weight_decay_max 1e-1 \
  --lambda_1_min 1e-5 \
  --lambda_1_max 1e-1 \
  --lambda_2_min 1e-5 \
  --lambda_2_max 1e-1 \
  --search_cancelout_init 1 \
  --cancelout_init 0.1
</code></pre>
    <h4>Docker equivalent: CancelOut + Optuna</h4>
    <pre class="card-pre"><code>mkdir -p result

docker run --rm \
  -v "$PWD/result:/results" \
  nolanzz/pipeline:latest \
  --input_path /app/data/ALLAML_10.npz \
  --out_dir /results \
  --method CANCELOUT \
  --name ALLAML_10_cancelout_optuna \
  --preprocess_mode external \
  --task_type binary \
  --is_snp 0 \
  --selected_activate sigmoid \
  --seed 0 \
  --use_evaluation 0 \
  --do_parameter_search 1 \
  --n_trials 40 \
  --n_jobs 4 \
  --use_cv 1 \
  --n_splits 5 \
  --lr_min 1e-5 \
  --lr_max 1e-2 \
  --dropout_min 0.2 \
  --dropout_max 0.7 \
  --weight_decay_min 0.0 \
  --weight_decay_max 1e-1 \
  --lambda_1_min 1e-5 \
  --lambda_1_max 1e-1 \
  --lambda_2_min 1e-5 \
  --lambda_2_max 1e-1 \
  --search_cancelout_init 1 \
  --cancelout_init 0.1
</code></pre>

    <!-- 6.3 eval, no hyper -->
    <h3>6.3 With evaluation, no hyperparameters</h3>
    <p>
      Generates “feature count vs performance” evaluation curves under fixed hyperparameters.
      See Section 4, <strong>Evaluation options</strong>, for independent test-set requirements and method-specific behavior.
    </p>
    <pre class="card-pre"><code>mkdir -p result

    apptainer run --bind "$(pwd)/result:/results" pipeline-cpu.sif \
      --input_path /app/data/ALLAML_10.npz \
      --out_dir /results \
      --method EARFS \
      --name ALLAML_10_earfs_evaluation \
      --preprocess_mode external \
      --task_type binary \
      --is_snp 0 \
      --selected_activate sigmoid \
      --seed 0 \
      --use_evaluation 1 \
      --do_parameter_search 0 \
      --max_features 20 \
      --feature_step 5
</code></pre>
    <h4>Docker equivalent</h4>
    <pre class="card-pre"><code>mkdir -p result

docker run --rm \
  -v "$PWD/result:/results" \
  nolanzz/pipeline:latest \
  --input_path /app/data/ALLAML_10.npz \
  --out_dir /results \
  --method EARFS \
  --name ALLAML_10_earfs_evaluation \
  --preprocess_mode external \
  --task_type binary \
  --is_snp 0 \
  --selected_activate sigmoid \
  --seed 0 \
  --use_evaluation 1 \
  --do_parameter_search 0 \
  --max_features 20 \
  --feature_step 5
</code></pre>
    <ul>
      <li>
        <strong>GRACES:</strong> use <code>--max_features_graces</code> to control the maximum number of features selected by GRACES
        and <code>--max_features</code> to control the maximum number of features evaluated.
        With evaluation enabled, <code>--max_features</code> must not be greater than
        <code>--max_features_graces</code> (<code>max_features &lt;= max_features_graces</code>).
      </li>
      <li>
        <strong>CAE:</strong> specify multiple feature counts with <code>--cat_k_select</code>. CAE runs separately for each K value.
        Unlike the ordinary evaluation sequence, CAE does not primarily construct K values from
        <code>--max_features</code> and <code>--feature_step</code>. These arguments, together with <code>--n_iters</code>,
        are still passed to the current CAE evaluation implementation, so their exact effect is implementation-dependent.
      </li>
    </ul>

    <!-- 6.4 eval, hyper -->
    <h3>6.4 With evaluation, with hyperparameters (Optuna + Evaluation)</h3>
    <p>First searches hyperparameters, then runs evaluation curves with the optimal configuration.</p>
    <pre class="card-pre"><code>mkdir -p result

    apptainer run --bind "$(pwd)/result:/results" pipeline-cpu.sif \
      --input_path /app/data/ALLAML_10.npz \
      --out_dir /results \
      --method GRACES \
      --name ALLAML_10_graces_full \
      --preprocess_mode external \
      --task_type binary \
      --is_snp 0 \
      --selected_activate sigmoid \
      --seed 0 \
      --use_evaluation 1 \
      --do_parameter_search 1 \
      --n_trials 40 \
      --n_jobs 4 \
      --use_cv 1 \
      --n_splits 5 \
      --eval_metric loss \
      --lr_min 1e-5 \
      --lr_max 1e-1 \
      --dropout_min 0.2 \
      --dropout_max 0.8 \
      --weight_decay_min 0.0 \
      --weight_decay_max 1e-1 \
      --alpha_min 0.85 \
      --alpha_max 0.99 \
      --f_correct_list 0,0.1,0.5,0.9 \
      --batch_size_list 8,16,32 \
      --digits_list 100,200,500 \
      --max_features 20 \
      --feature_step 5 \
      --max_features_graces 20
</code></pre>
    <h4>Docker equivalent</h4>
    <pre class="card-pre"><code>mkdir -p result

docker run --rm \
  -v "$PWD/result:/results" \
  nolanzz/pipeline:latest \
  --input_path /app/data/ALLAML_10.npz \
  --out_dir /results \
  --method GRACES \
  --name ALLAML_10_graces_full \
  --preprocess_mode external \
  --task_type binary \
  --is_snp 0 \
  --selected_activate sigmoid \
  --seed 0 \
  --use_evaluation 1 \
  --do_parameter_search 1 \
  --n_trials 40 \
  --n_jobs 4 \
  --use_cv 1 \
  --n_splits 5 \
  --eval_metric loss \
  --lr_min 1e-5 \
  --lr_max 1e-1 \
  --dropout_min 0.2 \
  --dropout_max 0.8 \
  --weight_decay_min 0.0 \
  --weight_decay_max 1e-1 \
  --alpha_min 0.85 \
  --alpha_max 0.99 \
  --f_correct_list 0,0.1,0.5,0.9 \
  --batch_size_list 8,16,32 \
  --digits_list 100,200,500 \
  --max_features 20 \
  --feature_step 5 \
  --max_features_graces 20
</code></pre>

    <ul>
      <li><strong>CancelOut (full mode):</strong> same as 6.2 plus <code>--use_evaluation 1</code>, <code>--max_features</code>, <code>--feature_step</code>.</li>
      <li><strong>Post-hoc explainers (DeepLIFT/GradientSHAP/FeatureAblation/Occlusion/LIME):</strong> support the same four scenarios; the main difference is the method choice (<code>--method</code>).</li>
    </ul>

    <!-- 6.5 CAE quick test -->
    <h3>6.5 CAE quick test</h3>
    <p>
      Runs CAE with three small feature-count settings suitable for a quick test with the bundled ALLAML dataset.
    </p>
    <pre class="card-pre"><code>mkdir -p result

    apptainer run --bind "$(pwd)/result:/results" pipeline-cpu.sif \
      --input_path /app/data/ALLAML_10.npz \
      --out_dir /results \
      --method CAE \
      --name ALLAML_10_cae_quick \
      --preprocess_mode external \
      --task_type binary \
      --is_snp 0 \
      --selected_activate sigmoid \
      --seed 0 \
      --use_evaluation 0 \
      --do_parameter_search 0 \
      --cat_k_select 5,10,20
</code></pre>
<h3>6.6 Running the same examples with GPU acceleration</h3>
<p>
  All Docker and Apptainer examples above use the CPU image by default. To run the same command with the published GPU
  image, keep the pipeline arguments unchanged and modify only the container image and GPU runtime flag as shown below.
</p>
<pre class="card-pre"><code># Apptainer GPU pattern
apptainer run --nv \
  --bind "$(pwd)/result:/results" \
  pipeline-gpu.sif \
  [pipeline arguments]

# Docker GPU pattern
docker run --rm --gpus all \
  -v "$PWD/result:/results" \
  nolanzz/pipeline:gpu \
  [pipeline arguments]</code></pre>
<p class="muted">
  GPU execution requires a compatible NVIDIA GPU and host driver. For Apptainer, use <code>--nv</code> so that the host
  NVIDIA libraries and devices are exposed inside the container. For Docker, install and configure the NVIDIA Container
  Toolkit and use <code>--gpus all</code>.
</p>
    <h4>Docker equivalent</h4>
    <pre class="card-pre"><code>mkdir -p result

docker run --rm \
  -v "$PWD/result:/results" \
  nolanzz/pipeline:latest \
  --input_path /app/data/ALLAML_10.npz \
  --out_dir /results \
  --method CAE \
  --name ALLAML_10_cae_quick \
  --preprocess_mode external \
  --task_type binary \
  --is_snp 0 \
  --selected_activate sigmoid \
  --seed 0 \
  --use_evaluation 0 \
  --do_parameter_search 0 \
  --cat_k_select 5,10,20
</code></pre>

  <h2 id="license">7 License</h2>
<p>
  This project is licensed under the
  <a href="https://www.gnu.org/licenses/gpl-3.0.html" target="_blank" rel="noopener noreferrer">GNU General Public License v3.0 (GPL-3.0)</a>.
  You may use, modify, and redistribute the software under the terms of the GPL-3.0 license.
</p>
