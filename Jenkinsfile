pipeline {
  agent {
    dockerfile {
        label 'isaac-gpu'
        reuseNode true
        filename 'Dockerfile.deps'
        args '-u root --gpus all -v /var/run/docker.sock:/var/run/docker.sock:rw'
    }
  }
  triggers {
    gitlab(triggerOnMergeRequest: true, branchFilterType: 'All')
  }
  stages {
    stage('Compile') {
      steps {
        sh '''mkdir -p nvblox/build'''
        sh '''mkdir -p nvblox/install'''
        sh '''cd nvblox/build && cmake .. -DCMAKE_INSTALL_PREFIX=../install && make -j8 && make install'''
      }
    }
    stage('Test') {
      steps {
        sh '''cd nvblox/build/tests && ctest -T test --no-compress-output'''
      }
    }
    stage('Link Into External Project') {
      steps {
        dir("nvblox_lib_test") {
          git credentialsId: 'isaac-git-master', url: 'ssh://git@gitlab-master.nvidia.com:12051/nvblox/nvblox_lib_test.git', branch: 'main'
        }
        sh '''mkdir -p nvblox_lib_test/build'''
        sh '''cd nvblox_lib_test/build && cmake .. -DNVBLOX_INSTALL_PATH=${WORKSPACE}/nvblox/install && make'''
        sh '''cd nvblox_lib_test/build && ./min_example'''
      }
    }
  }
  post {
    always {
      // Archive the CTest xml output
      archiveArtifacts (
        artifacts: 'nvblox/build/tests/Testing/**/*.xml',
        fingerprint: true
      )

      // Process the CTest xml output with the xUnit plugin
      xunit (
        testTimeMargin: '3000',
        thresholdMode: 1,
        thresholds: [
          skipped(failureThreshold: '0'),
          failed(failureThreshold: '0')
        ],
      tools: [CTest(
          pattern: 'nvblox/build/Testing/**/*.xml',
          deleteOutputFiles: true,
          failIfNotNew: false,
          skipNoTestFiles: true,
          stopProcessingIfError: true
        )]
      )

      // Clear the source and build dirs before next run
      cleanWs()
    }
    failure {
      updateGitlabCommitStatus name: 'build', state: 'failed'
    }
    success {
      updateGitlabCommitStatus name: 'build', state: 'success'
    }
  }
  options {
    gitLabConnection('gitlab-master')
  }
}
