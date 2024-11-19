#Packages
import numpy as np
import torch
import os
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional, Sigmoid
from tqdm import tqdm

#Complementary Scripts in Utils Folder
from .Utility.CTDS import DataSampler
from .Utility.CTDT import DataTransformer


'''
Main CTGAN script, output is synthetic conditional data
-------------------------------------------------------

purpose of this code is the create realistic synthetic data based on train data
that follows the same distributions of the real data by using a GAN foundation.

Code and its utils scripts are based upon the methodology of Xu et al. (2019)
DOI: https://doi.org/10.48550/arXiv.1907.00503
'''



class Generator(Module):
    """Generator for the CTGAN."""

    def __init__(self, noise_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = noise_dim
        layers = []
        for item in list(generator_dim):
            layers += [Residual(dim, item)]
            dim += item
        layers.append(Linear(dim, data_dim))
        self.seq = Sequential(*layers)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data
    

class Residual(Module):
    """Residual layer for the CTGAN. Used in Generator"""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input):
        """Apply the Residual layer to the `input`."""
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)


class Discriminator(Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        layers = []
        for item in list(discriminator_dim):
            layers += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        #output layer sigmoid (real or fake classification)
        layers += [Linear(dim, 1), Sigmoid()]
        #Construct the layers
        self.seq = Sequential(*layers)

        
    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
            """Compute the gradient penalty to improve the training stability and mitigate 
            common issues, such as mode collapse (lack of diversity), in GANs."""

            #renerate random interpolation coefficients
            alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
            alpha = alpha.repeat(1, pac, real_data.size(1))
            alpha = alpha.view(-1, real_data.size(1))

            #perform linear interpolation between real and fake data
            interpolates = alpha * real_data + ((1 - alpha) * fake_data)

            #Compute discriminator scores for interpolated samples
            disc_interpolates = self(interpolates)

            #Compute the gradients of D with respect to interpolated samples
            gradients = torch.autograd.grad(
                outputs=disc_interpolates, inputs=interpolates,
                grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]

            #compute, norm of gradients, and compute penalty as mean squared norm times lambda
            gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
            gradient_penalty = ((gradients_view) ** 2).mean() * lambda_
            return gradient_penalty


    def forward(self, input):
        """Apply the Discriminator to the `input`."""
        assert input.size()[0] % self.pac == 0
        return self.seq(input.view(-1, self.pacdim))


class CTGAN():
    
    def __init__(self, noise_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=True, epochs=30, pac=10, auto_save_models=False, 
                 model_folder='Models', type_data='User'):

        """
        Main CTGAN class: 
        
        This is the main CTGAN class. It combines the use of the Generator and Discriminator,
        along with various helper functions such as activation and loss calculation function.
        Finally both G and D are trained. 

        Args for CTGAN:
            noise_dim (int):
                Size of the random noise sample (latent space) passed to the Generator
            generator_dim (tuple):
                Size of the output samples for each one of the Residuals. A Residual Layer
                will be created for each one of the values provided.
            discriminator_dim (tuple):
                Size of the output samples for each one of the Discriminator Layers. A Linear Layer
                will be created for each one of the values provided.
            generator_lr (float):
                Learning rate for the generator.
            generator_decay (float):
                Generator weight decay for the Adam Optimizer.
            discriminator_lr (float):
                Learning rate for the discriminator.
            discriminator_decay (float):
                Discriminator weight decay for the Adam Optimizer.
            batch_size (int):
                Number of data samples to process in each step.
            discriminator_steps (int):
                Number of discriminator updates to do for each generator update.
                From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
                default is 5.
            log_frequency (boolean):
                Whether to use log frequency of categorical levels in conditional
                sampling. Defaults to True.
            verbose (boolean):
                Whether to have print statements for progress results.
            epochs (int):
                Number of training epochs.
            pac (int):
                Number of samples to group together when applying the discriminator.
                Defaults to 10.
            auto_save_models:
                Saves the trained D and G models in the default path, path can be set by 
                calling save/load helper function itself. Defaults to True
        """

        #Init variables
        self._noise_dim = noise_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac
        self.auto_save_models = auto_save_models
        self.model_folder = model_folder
        self.used_ids = set()
        self.type_data = type_data

        #Reset some components to mitigate error, when having two CTGAN
        self._generator = None
        self._discriminator = None
        self._transformer = None
        self._data_sampler = None

        #Check if batchsize is dividable
        assert batch_size % 2 == 0


    def _gumbel_softmax(self, logits, tau=1, hard=False, eps=1e-10, dim=-1, max_attempts=10):
        """Gumbel Softmax activation helper funciton for discrete variables, gumbel for stability"""

        #Tries 10 times to not get Nan values
        for _ in range(max_attempts):
            transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')


    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator (discrete or continuous)."""
        
        #Init empty list to store data, and set start index to 0
        data_t = []
        st = 0

        #For each column check discrete or continuos and apply correct activation fn
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                
                #Continous indeces
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                
                #Discrete indeces
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)


    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        #Stack individual losses and calculate avg loss
        loss = torch.stack(loss, dim=1)
        return (loss * m).sum() / data.size()[0]



    def _train_discriminator(self, train_data):
        '''Training Function for Discriminator, train on real data and generator output,
        the discriminator is fed fake data from the generator and real data. Basically, we are
        asking the discriminator if its able to tell us that this data is fake'''
        
        #Allow discriminator multiple training steps for one generator step (mitigates overpowering of G)
        for n in range(self._discriminator_steps):

            #Generate Noise and get conditional vector (Noise + Cond Vec will be input for Generator)
            noise = torch.normal(mean=self.noise_mean, std=self.noise_std)
            condvec = self._data_sampler.sample_condvec(self._batch_size)

            #Check if conditional vector is available
            if condvec is None:
                c1, m1, col, opt = None, None, None, None
                
                #Sample real data, with correct distributions, defined by sampler
                real = self._data_sampler.sample_data(train_data, self._batch_size, col, opt)
            else:
                c1, m1, col, opt = condvec
                c1 = torch.from_numpy(c1)
                m1 = torch.from_numpy(m1)
                #combine noise with conditional vector
                noise = torch.cat([noise, c1], dim=1) 

                #Shuffle data based on indeces
                perm = np.arange(self._batch_size)
                np.random.shuffle(perm)

                #Sample real data, with correct distributions, defined by sampler
                real = self._data_sampler.sample_data(train_data, self._batch_size, col[perm], opt[perm])
                c2 = c1[perm]

            #Generate fake data from the Generator based on noise + apply correct activation function
            fake = self._generator(noise)
            fakeact = self._apply_activate(fake)

            #Real data to tensor
            real = torch.from_numpy(real.astype('float32'))

            #Check if there is a conditional vector
            if c1 is not None:
                fake_cat = torch.cat([fakeact, c1], dim=1)
                real_cat = torch.cat([real, c2], dim=1)
            else:
                real_cat = real
                fake_cat = fakeact

            #Output of discriminator on real and on fake data
            y_fake = self._discriminator(fake_cat)
            y_real = self._discriminator(real_cat)

            #Calculate the penalty and calculate the loss
            pen = self._discriminator.calc_gradient_penalty(
                real_cat, fake_cat, pac= self.pac)
            loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

            #Do a backwards propagation and take D learning step
            self.optimizerD.zero_grad(set_to_none=False)
            pen.backward(retain_graph=True)
            loss_d.backward()
            self.optimizerD.step()

            #Return Loss for tracking (verbose)
            return loss_d.detach().item()


    def _training_generator(self):
        """Function for training the Generator: Pass generated samples to the discriminator 
        to assess its ability to classify them as real (assigned a label of 1). Compute the negative mean of the discriminator's 
        output as the loss. Minimize this loss to drive the generator to produce samples that maximize the discriminator's 
        output, aiming to approach a loss of 0. A loss near 0 indicates the discriminator increasingly assigns a label of 1 to 
        fake samples, resembling a perfect rate of "real" outputs for fake data. So no real data is used here"""

        #Generate Noise and get conditional vector (Noise + Cond Vec will later be input for Generator)
        noise = torch.normal(mean=self.noise_mean, std=self.noise_std)
        condvec = self._data_sampler.sample_condvec(self._batch_size)

        #Check if conditional vector is available
        if condvec is None:
                c1, m1, col, opt = None, None, None, None
        else:
                c1, m1, col, opt = condvec
                c1 = torch.from_numpy(c1)
                m1 = torch.from_numpy(m1)
                #combine noise with conditional vector
                noise = torch.cat([noise, c1], dim=1)

        #Generate Fake data with Generator and perform relevant activation functions to it
        fake = self._generator(noise)
        fakeact = self._apply_activate(fake)       

        #Check if conditional data is none, add it to fake data if its there
        if c1 is not None:
            y_fake = self._discriminator(torch.cat([fakeact, c1], dim=1))
        else:
            y_fake = self._discriminator(fakeact)


        #Check if conditional vector is none, other wise add loss for not compliying to conditional factors
        if condvec is None:
            cross_entropy = 0
        else:
            cross_entropy = self._cond_loss(fake, c1, m1)

        #Calculate loss as explained above
        loss_g = -torch.mean(y_fake) + cross_entropy

        #Reset Gradients of previous set (memory save) and take learning step
        self.optimizerG.zero_grad(set_to_none=False)
        loss_g.backward()
        self.optimizerG.step()

        #Return Loss of generator for verbose
        return loss_g.detach().item()


    def fit(self, train_data, discrete_columns=(), primary_key = None, trained_gen_model=None, trained_disc_model=None, id_column=None):
        """Fit function, train both the Discriminator and Generator. Insert real data
        and specify (list) the discrete columns in the dataset

        args:
        train_data          :   The real data on which will be trained (DataFrame)
        discrete_columns    :   List of discrete columns in the dataset
        primary_key         :   The name of primary key column in dataset (optional)
        trained_gen_model   :   Pre-trained generator model (Name of model in 'Saved Models' Folder) by default
        trained_disc_model  :   Pre-trained discriminator model (Name of model in 'Saved Models' Folder) by default
        id_field            :   Name of column which include ID's, for this new ID's will be created, when specified later (e.g. User Id's)
        """
        #Init Variable
        self.id_column = id_column

        #Load models if given
        if trained_disc_model is not None and trained_gen_model is not None:
            print(f'Fitting {self.type_data} CTGAN using pre-trained models')
            self._load_models(trained_disc_model,trained_gen_model)
        else:
            print("Training Models from Scratch")

        #Define the noise (latent space)
        self.noise_mean = torch.zeros(self._batch_size, self._noise_dim)
        self.noise_std = self.noise_mean + 1

        #init the CTGAN data transformer script
        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns, primary_key=primary_key)

        #Get transformed train data from transformer
        train_data = self._transformer.transform(train_data)

        #init the CTGAN data sampler script
        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency)

        # Get the dimensions of the transformed data (changed due to one hot encoding)
        data_dim = self._transformer.output_dimensions

        #Check if pre-trained models are provided else init new ones (Normal Training Process)
        if self._discriminator is None and self._generator is None:

            # Initialize the Generator, with noise dimensions and layers
            self._generator = Generator(
                self._noise_dim + self._data_sampler.dim_cond_vector(),  # Corrected method name
                self._generator_dim,
                data_dim
            )

            # Initialize the Discriminator with input dimensions and layers
            self._discriminator = Discriminator(
                data_dim + self._data_sampler.dim_cond_vector(),  # Corrected method name
                self._discriminator_dim,
                pac=self.pac
            )

            #init optimizer for Generator
            self.optimizerG = optim.Adam(
                self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
                weight_decay=self._generator_decay
            )

            #init optimizer for Discriminator
            self.optimizerD = optim.Adam(
                self._discriminator.parameters(), lr=self._discriminator_lr,
                betas=(0.5, 0.9), weight_decay=self._discriminator_decay
            )


            #Training Loop
            epoch_iterator = tqdm(range(self._epochs), disable=(not self._verbose))

            #Setup the description
            if self._verbose:
                description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
                epoch_iterator.set_description(description.format(gen=0, dis=0))

            #Calculate steps per epoch (batches)
            steps_per_epoch = max(len(train_data) // self._batch_size, 1)

            #Do the training loop for each epoch
            for i in epoch_iterator:
                for _ in range(steps_per_epoch):
                    d_loss = self._train_discriminator(train_data) #perform training D
                    g_loss = self._training_generator() #perform training G
                
                #Display the training information per epoch
                if self._verbose:
                    epoch_iterator.set_description(
                        description.format(gen=g_loss, dis=d_loss)
                    )
            
            #Auto Save the trained Models if set True
            if self.auto_save_models == True:
                self._save_models()
        

    def sample(self, n, condition_column=None, condition_value=None, new_ids=False):
        '''Sample function to generate new data, in N number of samples. Specific
        conditions are specified here'''
      
        #init variable
        self.new_ids = new_ids

        #Get the conditional vector of the specified condition column
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size)
        else:
            global_condition_vec = None

        #Get the data in batches
        steps = n // self._batch_size + 1
        data = []

        #Generate Noise, and get the global condvec or the normal condvec
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._noise_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            #Add condvec to the noise
            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1)
                noise = torch.cat([noise, c1], dim=1)

            #Generate fake data, with the (trained) Generator
            fake = self._generator(noise)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().numpy())

        #Concatenate the data and send to inverse transform
        data = np.concatenate(data, axis=0)
        data = data[:n]
        transformed_data = self._transformer.inverse_transform(data)

        #Change Ids to new ones
        if self.new_ids and self.id_column:
            new_ids = self._generate_new_ids(len(transformed_data))
            transformed_data[self.id_column] = new_ids

        return transformed_data
                   
    def _generate_new_ids(self, n):
        id_range = int(1e12)
        unique_ids = set()

        while len(unique_ids) < n:
            ids = np.random.randint(0, id_range, n - len(unique_ids), dtype=np.int64)
            unique_ids.update(id for id in ids if id not in self.used_ids)
        
        new_ids = list(unique_ids)
        self.used_ids.update(new_ids)
        return new_ids


    #Load Trained Models
    def _load_models(self, discriminator_model, generator_model):

        #Path
        gen_path = os.path.join(self.model_folder, generator_model)
        disc_path = os.path.join(self.model_folder, discriminator_model)

        #Load Models
        try:
            self._generator = torch.load(gen_path)
            self._discriminator = torch.load(disc_path)
        except Exception as e:
            raise RuntimeError ("Error Loading Models: ", e)
        
    #Save the Trained model
    def _save_models(self, name_discriminator=None, name_generator=None):
        
        #Set Names if not specified
        if name_generator is None:
            name_generator = 'CTGAN_Generator_model_'+str(self.type_data)+'.pth'
        if name_discriminator is None:
            name_discriminator = 'CTGAN_Discriminator_model_'+str(self.type_data)+'.pth'
        
        #Path
        gen_path = os.path.join(self.model_folder, name_generator)
        disc_path = os.path.join(self.model_folder, name_discriminator)

        #Save Generator & Discriminator
        if self._discriminator is None or self._generator is None:
            raise ValueError (f'Generator and Discriminator not trained yet')
        else:
            torch.save(self._generator, gen_path)
            torch.save(self._discriminator, disc_path)
