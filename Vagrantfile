# ==============================================================================
# = project       :- pysci-tutorial
# = module        :- Vagrantfile
# = author        :- sameh kamal
# = description   :- vagrant configuration file
# ==============================================================================
Vagrant.configure(2) do |config|
  # use ubuntu 14.4.0 64bit
  config.vm.box = "ubuntu/trusty64"

  # configure machine memory to be 2GBs
  config.vm.provider "virtualbox" do |vb|
    vb.memory = "2048"
  end
  
  # use shell provisioning from external script
  config.vm.provision "shell", path: "./provision/bootstrap.sh"
end
