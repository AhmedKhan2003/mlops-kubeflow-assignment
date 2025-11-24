pipeline {
  agent any
  stages {
    stage('Checkout') {
      steps {
        checkout scm
      }
    }
    stage('Setup Python') {
      steps {
        sh 'python3 -m venv .venv'
        sh '. .venv/bin/activate && pip install -r requirements.txt'
      }
    }
    stage('DVC Status') {
      steps {
        sh '. .venv/bin/activate && dvc status'
      }
    }
    stage('Compile Pipeline') {
      steps {
        sh '. .venv/bin/activate && python src/compile_components.py && python pipeline.py'
      }
    }
  }
  post {
    success {
      echo 'CI pipeline passed'
    }
    failure {
      echo 'CI pipeline failed'
    }
  }
}
