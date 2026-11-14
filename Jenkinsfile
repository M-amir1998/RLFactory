// RLFactory CPU-only CI. No GPU, CUDA, model downloads, Hugging Face authentication or paid APIs.
// Stages run sequentially (16 GB RAM agents): heavyweight tests are never parallelised.
pipeline {
    agent { label 'linux' }

    options {
        disableConcurrentBuilds()
        timestamps()
        timeout(time: 60, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '20', artifactNumToKeepStr: '10'))
    }

    parameters {
        booleanParam(name: 'RUN_DOCKER_BUILD', defaultValue: true, description: 'Build the CLI and sandbox images')
        booleanParam(name: 'RUN_PIP_AUDIT', defaultValue: false, description: 'Query public advisory databases (needs network)')
    }

    environment {
        UV_CACHE_DIR = "${WORKSPACE}/.uv-cache"
        UV_PYTHON = '3.12'
        UV_NO_PROGRESS = '1'
        HF_HUB_OFFLINE = '1'
        TRANSFORMERS_OFFLINE = '1'
        HF_HUB_DISABLE_TELEMETRY = '1'
        CUDA_VISIBLE_DEVICES = ''
        RLFACTORY_RESULTS_DIR = "${WORKSPACE}/results"
        RLFACTORY_DISK_RESERVE_GB = '1'
        PYTEST_FLAGS = '-p no:cacheprovider --cov=rlfactory --cov-append --cov-report='
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                sh 'git log -1 --format="%h %s"'
                sh 'rm -rf reports/junit reports/coverage.xml .coverage dist results'
            }
        }
        stage('Dependency validation') {
            steps {
                sh 'uv --version'
                sh 'uv lock --check'
                sh 'uv sync --frozen'
                sh 'uv pip check'
            }
        }
        stage('Formatting') {
            steps { sh 'uv run ruff format --check .' }
        }
        stage('Ruff') {
            steps { sh 'uv run ruff check .' }
        }
        stage('Strict mypy') {
            steps { sh 'uv run mypy' }
        }
        stage('Unit tests') {
            steps {
                sh 'uv run pytest tests/unit -m "not sandbox and not verifier and not store and not docker" --junitxml=reports/junit/unit.xml $PYTEST_FLAGS'
            }
        }
        stage('Property tests') {
            steps { sh 'uv run pytest tests/property --junitxml=reports/junit/property.xml $PYTEST_FLAGS' }
        }
        stage('Integration tests') {
            steps {
                sh 'uv run pytest tests/integration -m "not trl and not tiny_model" --junitxml=reports/junit/integration.xml $PYTEST_FLAGS'
            }
        }
        stage('Environment sandbox tests') {
            steps {
                // Docker-marked tests self-skip when the daemon or sandbox image is unavailable.
                sh 'uv run pytest tests/unit -m "sandbox and not verifier" --junitxml=reports/junit/sandbox.xml $PYTEST_FLAGS'
            }
        }
        stage('Verifier tests') {
            steps { sh 'uv run pytest tests/unit -m verifier --junitxml=reports/junit/verifiers.xml $PYTEST_FLAGS' }
        }
        stage('Trajectory-store tests') {
            steps { sh 'uv run pytest tests/unit -m store --junitxml=reports/junit/store.xml $PYTEST_FLAGS' }
        }
        stage('Mock rollout E2E') {
            steps { sh 'uv run pytest tests/e2e/test_rollout_e2e.py --junitxml=reports/junit/e2e-rollout.xml $PYTEST_FLAGS' }
        }
        stage('Mock post-training E2E') {
            steps {
                sh 'uv run pytest tests/e2e/test_cpu_e2e.py --junitxml=reports/junit/e2e-post-training.xml $PYTEST_FLAGS'
                sh 'uv run rlfactory experiment run configs/experiments/ci-e2e.yaml --run-id "ci-e2e-${BUILD_NUMBER}"'
                sh 'uv run coverage xml -o reports/coverage.xml'
                sh 'uv run coverage report --fail-under=85'
            }
        }
        stage('Security scan') {
            steps {
                sh 'uv run python scripts/secret_scan.py'
                sh 'uv run python scripts/check_repo_hygiene.py'
                script {
                    if (params.RUN_PIP_AUDIT) {
                        sh 'uv export --frozen --no-dev --format requirements-txt > reports/requirements.txt'
                        sh 'uvx pip-audit --strict -r reports/requirements.txt'
                    }
                }
            }
        }
        stage('Package build') {
            steps {
                sh 'uv build'
                sh 'uv run python scripts/check_repo_hygiene.py --dist dist'
            }
        }
        stage('Docker build') {
            when { expression { return params.RUN_DOCKER_BUILD } }
            steps {
                sh 'docker build --build-arg RLFACTORY_GIT_SHA="$(git rev-parse HEAD)" -t "rlfactory:ci-${BUILD_NUMBER}" .'
                sh 'docker build -f docker/sandbox.Dockerfile -t "rlfactory-sandbox:ci-${BUILD_NUMBER}" docker'
                sh 'docker run --rm "rlfactory:ci-${BUILD_NUMBER}" doctor --no-docker'
            }
        }
    }

    post {
        always {
            junit allowEmptyResults: true, testResults: 'reports/junit/*.xml'
            // Small artefacts only: never rollouts/, checkpoints, datasets or model weights.
            archiveArtifacts allowEmptyArchive: true, fingerprint: true, artifacts: [
                'reports/junit/*.xml',
                'reports/coverage.xml',
                'dist/*.whl',
                'dist/*.tar.gz',
                'results/ci-e2e-*/report.md',
                'results/ci-e2e-*/plots/*.svg',
                'results/ci-e2e-*/manifest.json',
            ].join(', ')
        }
        cleanup {
            sh 'rm -rf results dist'
            sh 'docker image rm "rlfactory:ci-${BUILD_NUMBER}" "rlfactory-sandbox:ci-${BUILD_NUMBER}" >/dev/null 2>&1 || true'
        }
    }
}
