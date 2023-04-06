pipeline {
  agent none
  triggers {
    gitlab(triggerOnMergeRequest: true, branchFilterType: 'All')
  }
  stages {
    stage("Compile & Test Multiplatform") {
      parallel {
        stage("x86") {
          agent {
            dockerfile {
                label 'isaac-gpu'
                reuseNode true
                filename 'docker/Dockerfile.deps'
                args '-u root --gpus all -v /var/run/docker.sock:/var/run/docker.sock:rw'
            }
          }
          stages {
            stage('Compile x86') {
              steps {
                sh '''mkdir -p nvblox/build'''
                sh '''mkdir -p nvblox/install'''
                sh '''cd nvblox/build && cmake .. -DCMAKE_INSTALL_PREFIX=../install && make clean && make -j8 && make install'''
              }
            }
            stage('Lint') {
              steps {
                sh '''bash nvblox/lint/lint_nvblox_h.sh'''
              }
            }
            stage('Test x86') {
              steps {
                sh '''cd nvblox/build/tests && ctest -T test --no-compress-output'''
              }
            }
            stage('Link Into External Project x86') {
              steps {
                dir("nvblox_lib_test") {
                  git credentialsId: 'vault-svc-ssh', url: 'ssh://git@gitlab-master.nvidia.com:12051/nvblox/nvblox_lib_test.git', branch: 'main'
                }
                sh '''mkdir -p nvblox_lib_test/build'''
                sh '''cd nvblox_lib_test/build && cmake .. -DNVBLOX_INSTALL_PATH=${WORKSPACE}/nvblox/install && make'''
                sh '''cd nvblox_lib_test/build && ./min_example'''
              }
            }
            stage("Cleanup x86") {
              steps {
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
                    pattern: 'nvblox/build/tests/Testing/**/*.xml',
                    deleteOutputFiles: true,
                    failIfNotNew: false,
                    skipNoTestFiles: true,
                    stopProcessingIfError: true
                  )]
                )

                // Clear the source and build dirs before next run
                cleanWs()
              }
            }
          }
        }
        stage("Jetson 5.1.1") {
          agent {
            dockerfile {
                label 'jp-5.1.1'
                reuseNode true
                filename 'docker/Dockerfile.jetson_deps'
                args '-u root --runtime nvidia --gpus all -v /var/run/docker.sock:/var/run/docker.sock:rw'
            }
          }
          stages {
            stage('Compile Jetson') {
              steps {
                sh '''mkdir -p nvblox/build'''
                sh '''mkdir -p nvblox/install'''
                sh '''cd nvblox/build && cmake .. -DCMAKE_INSTALL_PREFIX=../install && make clean && make -j8 && make install'''
              }
            }
            stage('Lint') {
              steps {
                sh '''bash nvblox/lint/lint_nvblox_h.sh'''
              }
            }
            stage('Test Jetson') {
              steps {
                sh '''cd nvblox/build/tests && ctest -T test --no-compress-output'''
              }
            }
            stage('Link Into External Project Jetson') {
              steps {
                dir("nvblox_lib_test") {
                  git credentialsId: 'vault-svc-ssh', url: 'ssh://git@gitlab-master.nvidia.com:12051/nvblox/nvblox_lib_test.git', branch: 'main'
                }
                sh '''mkdir -p nvblox_lib_test/build'''
                sh '''cd nvblox_lib_test/build && cmake .. -DNVBLOX_INSTALL_PATH=${WORKSPACE}/nvblox/install && make'''
                sh '''cd nvblox_lib_test/build && ./min_example'''
              }
            }
            stage("Cleanup Jetson") {
              steps {
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
                    pattern: 'nvblox/build/tests/Testing/**/*.xml',
                    deleteOutputFiles: true,
                    failIfNotNew: false,
                    skipNoTestFiles: true,
                    stopProcessingIfError: true
                  )]
                )

                // Clear the source and build dirs before next run
                cleanWs()
              }
            }
          }
        }
      }
    }
  }
  post {
    always {
      agent {
        label 'isaac-gpu'
      }
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
