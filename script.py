import subprocess
# Clean up the train_dir for log files
subprocess.call(['rm', '-rf', './train_dir'])

# Run all the trainings {relu, lrelu, selu}
subprocess.Popen(['python', 'trainer.py', '--activation', 'relu'])
subprocess.Popen(['python', 'trainer.py', '--activation', 'lrelu'])
subprocess.Popen(['python', 'trainer.py', '--activation', 'selu'])

# Run Tensorboard
subprocess.call(['sleep', '10']) 
subprocess.call(['python', 'monitor.py', 'train_dir/', '--port', '7007'])
