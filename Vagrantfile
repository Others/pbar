Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/eoan64"

  config.vm.boot_timeout = 1000

  # Disable the default syncing
  config.vm.synced_folder ".", "/vagrant", disabled: true
  # Instead sync the actual worker folder and the lib if it's around
  config.vm.synced_folder ".", "/home/vagrant/pbar"

  config.vm.provider :virtualbox do |vb|
    vb.customize ["modifyvm", :id, "--ioapic", "on"] 
    vb.customize ["modifyvm", :id, "--memory", "4096"]
    vb.customize ["modifyvm", :id, "--cpus", "2"]
    vb.customize ["modifyvm", :id, "--uartmode1", "disconnected" ]
    vb.customize ["modifyvm", :id, "--natdnshostresolver1", "on"]
  end

  # timezone = 'America/New York'

  config.vm.provision "shell", privileged: false, inline: <<-SHELL
    sudo apt-get update
    sudo apt-get upgrade -y
    sudo apt-get autoremove -y
    sudo apt-get install python3 python3-dev python3-pip -y
    sudo apt-get install vim git tmux -y
    sudo apt-get install httpie -y
    sudo apt-get install gfortran -y
    sudo apt-get install libblas3 liblapack3 liblapack-dev libblas-dev -y
    sudo apt-get update
    sudo apt-get upgrade -y
    sudo apt-get autoremove -y
 SHELL
end

