import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import networkx as nx

class CorrelationAnalysis:
    """
    Tools for analyzing correlations between assets.
    """
    
    @staticmethod
    def calculate_correlation_matrix(returns_df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
        """
        Calculate correlation matrix for asset returns.
        
        Args:
            returns_df: DataFrame of asset returns (each column is an asset)
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            
        Returns:
            Correlation matrix
        """
        valid_methods = ['pearson', 'spearman', 'kendall']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr(method=method)
        
        return corr_matrix
    
    @staticmethod
    def calculate_rolling_correlation(returns_1: pd.Series, returns_2: pd.Series, 
                                    window: int = 60, method: str = 'pearson') -> pd.Series:
        """
        Calculate rolling correlation between two assets.
        
        Args:
            returns_1: Returns series for first asset
            returns_2: Returns series for second asset
            window: Rolling window size in days
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            
        Returns:
            Series of rolling correlations
        """
        # Align series (handle missing values)
        aligned_df = pd.DataFrame({'asset1': returns_1, 'asset2': returns_2})
        aligned_df = aligned_df.dropna()
        
        # Calculate rolling correlation
        rolling_corr = aligned_df['asset1'].rolling(window=window).corr(aligned_df['asset2'], method=method)
        
        return rolling_corr
    
    @staticmethod
    def find_highest_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
        """
        Find pairs with highest correlations.
        
        Args:
            corr_matrix: Correlation matrix
            threshold: Minimum correlation to include
            
        Returns:
            DataFrame with sorted pairs and correlations
        """
        # Create list of pairs and correlations
        pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                asset1 = corr_matrix.columns[i]
                asset2 = corr_matrix.columns[j]
                corr = corr_matrix.iloc[i, j]
                
                if abs(corr) >= threshold:
                    pairs.append({
                        'asset1': asset1,
                        'asset2': asset2,
                        'correlation': corr
                    })
        
        # Convert to DataFrame and sort
        if pairs:
            pairs_df = pd.DataFrame(pairs)
            pairs_df = pairs_df.sort_values('correlation', ascending=False)
            return pairs_df
        else:
            return pd.DataFrame(columns=['asset1', 'asset2', 'correlation'])
    
    @staticmethod
    def find_lowest_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
        """
        Find pairs with lowest correlations (for diversification).
        
        Args:
            corr_matrix: Correlation matrix
            threshold: Maximum correlation to include
            
        Returns:
            DataFrame with sorted pairs and correlations
        """
        # Create list of pairs and correlations
        pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                asset1 = corr_matrix.columns[i]
                asset2 = corr_matrix.columns[j]
                corr = corr_matrix.iloc[i, j]
                
                if abs(corr) <= threshold:
                    pairs.append({
                        'asset1': asset1,
                        'asset2': asset2,
                        'correlation': corr
                    })
        
        # Convert to DataFrame and sort
        if pairs:
            pairs_df = pd.DataFrame(pairs)
            pairs_df = pairs_df.sort_values('correlation', ascending=True)
            return pairs_df
        else:
            return pd.DataFrame(columns=['asset1', 'asset2', 'correlation'])
    
    @staticmethod
    def test_correlation_significance(returns_1: pd.Series, returns_2: pd.Series, 
                                    method: str = 'pearson') -> Dict[str, float]:
        """
        Test statistical significance of correlation.
        
        Args:
            returns_1: Returns series for first asset
            returns_2: Returns series for second asset
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            
        Returns:
            Dictionary with correlation coefficient and p-value
        """
        # Align series (handle missing values)
        aligned_df = pd.DataFrame({'asset1': returns_1, 'asset2': returns_2})
        aligned_df = aligned_df.dropna()
        
        # Calculate correlation and p-value
        if method == 'pearson':
            corr, p_value = stats.pearsonr(aligned_df['asset1'], aligned_df['asset2'])
        elif method == 'spearman':
            corr, p_value = stats.spearmanr(aligned_df['asset1'], aligned_df['asset2'])
        elif method == 'kendall':
            corr, p_value = stats.kendalltau(aligned_df['asset1'], aligned_df['asset2'])
        else:
            raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'")
        
        return {
            'correlation': corr,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'method': method
        }
    
    @staticmethod
    def create_correlation_network(corr_matrix: pd.DataFrame, threshold: float = 0.7) -> nx.Graph:
        """
        Create a network graph from correlation matrix.
        
        Args:
            corr_matrix: Correlation matrix
            threshold: Minimum absolute correlation to create an edge
            
        Returns:
            NetworkX graph object
        """
        # Create an empty graph
        G = nx.Graph()
        
        # Add nodes (assets)
        for asset in corr_matrix.columns:
            G.add_node(asset)
        
        # Add edges (correlations)
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                asset1 = corr_matrix.columns[i]
                asset2 = corr_matrix.columns[j]
                corr = corr_matrix.iloc[i, j]
                
                if abs(corr) >= threshold:
                    G.add_edge(asset1, asset2, weight=corr)
        
        return G


class CovarianceAnalysis:
    """
    Tools for analyzing covariance of asset returns.
    """
    
    @staticmethod
    def calculate_covariance_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate covariance matrix for asset returns.
        
        Args:
            returns_df: DataFrame of asset returns (each column is an asset)
            
        Returns:
            Covariance matrix
        """
        # Calculate covariance matrix
        cov_matrix = returns_df.cov()
        
        return cov_matrix
    
    @staticmethod
    def calculate_rolling_covariance(returns_1: pd.Series, returns_2: pd.Series, 
                                   window: int = 60) -> pd.Series:
        """
        Calculate rolling covariance between two assets.
        
        Args:
            returns_1: Returns series for first asset
            returns_2: Returns series for second asset
            window: Rolling window size in days
            
        Returns:
            Series of rolling covariances
        """
        # Align series (handle missing values)
        aligned_df = pd.DataFrame({'asset1': returns_1, 'asset2': returns_2})
        aligned_df = aligned_df.dropna()
        
        # Calculate rolling covariance
        rolling_cov = aligned_df['asset1'].rolling(window=window).cov(aligned_df['asset2'])
        
        return rolling_cov
    
    @staticmethod
    def calculate_ewma_covariance(returns_df: pd.DataFrame, span: int = 30) -> pd.DataFrame:
        """
        Calculate exponentially weighted moving average covariance matrix.
        
        Args:
            returns_df: DataFrame of asset returns (each column is an asset)
            span: Span for exponential weighting
            
        Returns:
            EWMA covariance matrix
        """
        # Calculate EWMA covariance matrix
        ewma_cov = returns_df.ewm(span=span).cov()
        
        # Reshape the result to a proper covariance matrix
        assets = returns_df.columns
        periods = ewma_cov.index.levels[0]
        
        # Get the last period
        last_period = periods[-1]
        
        # Extract the covariance matrix for the last period
        last_cov = ewma_cov.loc[last_period]
        
        return last_cov


class PrincipalComponentAnalysis:
    """
    Principal Component Analysis for asset returns.
    """
    
    @staticmethod
    def perform_pca(returns_df: pd.DataFrame, n_components: Optional[int] = None) -> Dict[str, any]:
        """
        Perform Principal Component Analysis on asset returns.
        
        Args:
            returns_df: DataFrame of asset returns (each column is an asset)
            n_components: Number of components to extract (default: None for all)
            
        Returns:
            Dictionary with PCA results
        """
        from sklearn.decomposition import PCA
        
        # Fill missing values with column mean
        returns_filled = returns_df.fillna(returns_df.mean())
        
        # Create and fit PCA model
        pca = PCA(n_components=n_components)
        pca.fit(returns_filled)
        
        # Transform the data
        transformed_data = pca.transform(returns_filled)
        
        # Create DataFrame of transformed data
        transformed_df = pd.DataFrame(
            transformed_data, 
            index=returns_df.index,
            columns=[f'PC{i+1}' for i in range(transformed_data.shape[1])]
        )
        
        # Create DataFrame of component loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            index=returns_df.columns,
            columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])]
        )
        
        # Calculate explained variance ratio
        explained_var = pd.Series(
            pca.explained_variance_ratio_,
            index=[f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))]
        )
        
        # Cumulative explained variance
        cumulative_var = explained_var.cumsum()
        
        return {
            'pca': pca,
            'transformed_data': transformed_df,
            'loadings': loadings,
            'explained_variance': explained_var,
            'cumulative_variance': cumulative_var
        }
    
    @staticmethod
    def get_top_contributors(pca_result: Dict[str, any], component: int = 1, 
                           top_n: int = 5) -> pd.DataFrame:
        """
        Get the top contributors to a principal component.
        
        Args:
            pca_result: Result from perform_pca
            component: Component number (1-based)
            top_n: Number of top contributors to return
            
        Returns:
            DataFrame with top contributors
        """
        # Get the loadings DataFrame
        loadings = pca_result['loadings']
        
        # Component name
        component_name = f'PC{component}'
        
        if component_name not in loadings.columns:
            raise ValueError(f"Component {component} not found in PCA results")
        
        # Get absolute loadings for the component
        abs_loadings = loadings[component_name].abs()
        
        # Sort and get top contributors
        top_contributors = abs_loadings.sort_values(ascending=False).head(top_n)
        
        # Create result DataFrame with original signs
        result = pd.DataFrame({
            'Asset': top_contributors.index,
            'Loading': [loadings.loc[asset, component_name] for asset in top_contributors.index],
            'Absolute Loading': top_contributors.values
        })
        
        return result 