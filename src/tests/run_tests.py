from verify_setup import verify_setup
from test_dataloading import test_dataloaders

def main():
    print("Running setup verification...")
    verify_setup()
    
    print("\nRunning dataloader tests...")
    test_dataloaders()

if __name__ == "__main__":
    main()