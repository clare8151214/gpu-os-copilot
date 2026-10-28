pipeline {
    agent { label 'linux-ci' }
    options {
        timestamps()
        timeout(time: 15, unit: 'MINUTES')
        disableConcurrentBuilds()
        skipDefaultCheckout(true)
    }
    environment {
        PIP_DISABLE_PIP_VERSION_CHECK = '1'
        PYTHONDONTWRITEBYTECODE = '1'
        DRY_RUN = '1'
    }
    stages {
        stage('Checkout') {
            steps {
                checkout scm
                sh 'git rev-parse --short HEAD'
            }
        }
        stage('Setup Python') {
            steps {
                sh '''
                    python3 -m venv --clear .venv
                    .venv/bin/python -m pip install -c requirements-dev.lock -e ".[dev,meta]"
                '''
            }
        }
        stage('Static checks') {
            steps {
                sh '''
                    .venv/bin/ruff check .
                    .venv/bin/ruff format --check .
                '''
            }
        }
        stage('Unit tests') {
            steps {
                sh '.venv/bin/python -m pytest tests/unit -q --junitxml=test-results/unit.xml'
            }
        }
        stage('Offline integration tests') {
            steps {
                sh '.venv/bin/python -m pytest tests/integration -q --junitxml=test-results/integration.xml'
            }
        }
    }
    post {
        always {
            junit allowEmptyResults: true, testResults: 'test-results/*.xml'
        }
    }
}
