import torch

from torch import nn
from torch.nn import functional
from torch.utils.data import TensorDataset, DataLoader

class ExplicitComponent(nn.Module):

    def __init__(self, in_size, out_size, num_layers):
        super(ExplicitComponent, self).__init__()
        self._layer0 = nn.Linear(in_size, out_size, bias=False)  # make sum of 1st-order interactions
        self._layers = nn.ModuleList([nn.Linear(out_size, out_size) for _ in range(num_layers)])

    def forward(self, x):

        interactions = []

        x0 = self._layer0(x)
        xi = x0

        for layer in self._layers:
            xi = layer(xi * x0)
            interactions.append(xi.sum(dim=2))

        return torch.concat(interactions, dim=1)

class ImplicitComponent(nn.Module):

    def __init__(self, in_size, out_size, num_layers):
        super(ImplicitComponent, self).__init__()
        self._layers = nn.ModuleList([nn.Linear(in_size, out_size)])
        for _ in range(num_layers - 1):
            self._layers.append(nn.Linear(out_size, out_size))

    def forward(self, x):
        
        interactions = []

        xi = x

        for layer in self._layers:
            xi = layer(xi)
            xi = functional.leaky_relu(xi)
            interactions.append(xi.sum(dim=2))

        return torch.concat(interactions, dim=1)

class XIN(nn.Module):

    def __init__(
        self,
        num_numerical_x, num_categorical_x, num_categories_in_x, out_size, task,
        embedding_size=8
    ) -> None:

        super(XIN, self).__init__()

        self._task = task

        self._explicit_numerical_x_embedding_layers = nn.ModuleList([nn.Linear(1, embedding_size, bias=False) for _ in range(num_numerical_x)])
        self._explicit_categorical_x_embedding_layers = nn.ModuleList([nn.Embedding(num_categories_in_x[i] + 1, embedding_size, padding_idx=num_categories_in_x[i]) for i in range(num_categorical_x)])

        self._implicit_numerical_x_embedding_layers = nn.ModuleList([nn.Linear(1, embedding_size, bias=False) for _ in range(num_numerical_x)])
        self._implicit_categorical_x_embedding_layers = nn.ModuleList([nn.Embedding(num_categories_in_x[i] + 1, embedding_size, padding_idx=num_categories_in_x[i]) for i in range(num_categorical_x)])

        self._explicit_component = ExplicitComponent(num_numerical_x + num_categorical_x, embedding_size, num_layers=2)
        self._implicit_component = ImplicitComponent(num_numerical_x + num_categorical_x, embedding_size, num_layers=2)
        
        self._output_layer = nn.Linear(2 * (num_numerical_x + num_categorical_x) * embedding_size + (2 + 2) * embedding_size, out_size)

        self.init_parameters()

    def init_parameters(self):
        for layer in self._explicit_numerical_x_embedding_layers:
            nn.init.uniform_(layer.weight, -1e-4, 1e-4)
        for layer in self._implicit_numerical_x_embedding_layers:
            nn.init.uniform_(layer.weight, -1e-4, 1e-4)
        for layer in self._explicit_categorical_x_embedding_layers:
            nn.init.uniform_(layer.weight[:-1], -1e-4, 1e-4)  # initialize without padding idx
        for layer in self._implicit_categorical_x_embedding_layers:
            nn.init.uniform_(layer.weight[:-1], -1e-4, 1e-4)  # initialize without padding idx

    def forward(self, numerical_x, categorical_x):

        explicit_numerical_x_embeddings = [embedding_layer(numerical_x[:, i, None]) for i, embedding_layer in enumerate(self._explicit_numerical_x_embedding_layers)]
        explicit_categorical_x_embeddings = [embedding_layer(categorical_x[:, i]) for i, embedding_layer in enumerate(self._explicit_categorical_x_embedding_layers)]

        explicit_numerical_x_embeddings_stack = torch.stack(explicit_numerical_x_embeddings, dim=1)
        explicit_categorical_x_embeddings_stack = torch.stack(explicit_categorical_x_embeddings, dim=1)
        explicit_x_stack = torch.concat([explicit_numerical_x_embeddings_stack, explicit_categorical_x_embeddings_stack], dim=1).transpose(1, 2)

        explicit_numerical_x_embeddings_concat = torch.concat(explicit_numerical_x_embeddings, dim=1)
        explicit_categorical_x_embeddings_concat = torch.concat(explicit_categorical_x_embeddings, dim=1)
        explicit_x_concat = torch.concat([explicit_numerical_x_embeddings_concat, explicit_categorical_x_embeddings_concat], dim=1)

        implicit_numerical_x_embeddings = [embedding_layer(numerical_x[:, i, None]) for i, embedding_layer in enumerate(self._implicit_numerical_x_embedding_layers)]
        implicit_categorical_x_embeddings = [embedding_layer(categorical_x[:, i]) for i, embedding_layer in enumerate(self._implicit_categorical_x_embedding_layers)]

        implicit_numerical_x_embeddings_stack = torch.stack(implicit_numerical_x_embeddings, dim=1)
        implicit_categorical_x_embeddings_stack = torch.stack(implicit_categorical_x_embeddings, dim=1)
        implicit_x_stack = torch.concat([implicit_numerical_x_embeddings_stack, implicit_categorical_x_embeddings_stack], dim=1).transpose(1, 2)

        implicit_numerical_x_embeddings_concat = torch.concat(implicit_numerical_x_embeddings, dim=1)
        implicit_categorical_x_embeddings_concat = torch.concat(implicit_categorical_x_embeddings, dim=1)
        implicit_x_concat = torch.concat([implicit_numerical_x_embeddings_concat, implicit_categorical_x_embeddings_concat], dim=1)

        explicit_interactions = self._explicit_component(explicit_x_stack)
        implicit_interactions = self._implicit_component(implicit_x_stack)

        interactions = torch.concat([explicit_x_concat, implicit_x_concat, explicit_interactions, implicit_interactions], dim=1)

        output = self._output_layer(interactions)
        if self._task == 'binary_classification':
            output = torch.sigmoid(output)
        elif self._task == 'classification':
            output = torch.softmax(output)

        return output

    def fit(
        self,
        train_numerical_x, train_categorical_x, train_y,
        val_numerical_x, val_categorical_x, val_y,
        lr=1e-3, batch_size=128, epochs=1000, num_workers=0
    ):

        device = next(self.parameters()).device

        train_numerical_x = torch.tensor(train_numerical_x, dtype=torch.float32)
        train_categorical_x = torch.tensor(train_categorical_x, dtype=torch.int32)
        train_y = torch.tensor(train_y, dtype=torch.float32)
        train_dataset = TensorDataset(train_numerical_x, train_categorical_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
        
        val_numerical_x = torch.tensor(val_numerical_x, dtype=torch.float32)
        val_categorical_x = torch.tensor(val_categorical_x, dtype=torch.int32)
        val_y = torch.tensor(val_y, dtype=torch.float32)
        val_dataset = TensorDataset(val_numerical_x, val_categorical_x, val_y)
        val_loader = DataLoader(val_dataset, batch_size, num_workers=num_workers)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if self._task == 'regression':
            criterion = functional.mse_loss
        elif self._task == 'binary_classification':
            criterion = functional.binary_cross_entropy
        else:
            criterion = functional.cross_entropy

        for epoch in range(1, epochs + 1):

            train_loss, train_step = 0, 0
            val_loss, val_step = 0, 0

            self.train()

            for batch_numerical_x, batch_categorical_x, batch_y in train_loader:

                batch_numerical_x = batch_numerical_x.to(device)
                batch_categorical_x = batch_categorical_x.to(device)
                batch_y = batch_y.to(device)

                output = self(batch_numerical_x, batch_categorical_x)
                loss = criterion(output, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_step += 1

            with torch.no_grad():

                self.eval()

                for batch_numerical_x, batch_categorical_x, batch_y in val_loader:

                    batch_numerical_x = batch_numerical_x.to(device)
                    batch_categorical_x = batch_categorical_x.to(device)
                    batch_y = batch_y.to(device)

                    output = self(batch_numerical_x, batch_categorical_x)
                    loss = criterion(output, batch_y)

                    val_loss += loss.item()
                    val_step += 1

            train_loss /= train_step
            val_loss /= val_step

            print(f'Epoch {epoch:>5}, Train Loss: {train_loss/train_step:.7f}, Val Loss: {val_loss/val_step:.7f}')

    def predict(self, numerical_x, categorical_x, batch_size=128, num_workers=0):

        device = next(self.parameters()).device

        test_numerical_x = torch.tensor(numerical_x, dtype=torch.float32)
        test_categorical_x = torch.tensor(categorical_x, dtype=torch.int32)
        test_dataset = TensorDataset(test_numerical_x, test_categorical_x)
        test_loader = DataLoader(test_dataset, batch_size, num_workers=num_workers)

        total_output = []

        with torch.no_grad():

            for batch_numerical_x, batch_categorical_x in test_loader:

                batch_numerical_x = batch_numerical_x.to(device)
                batch_categorical_x = batch_categorical_x.to(device)

                output = self(batch_numerical_x, batch_categorical_x).detach().cpu().numpy()
                total_output.extend(output)

        return total_output
