
import sys, os, time, gc, json
import torch
from torch.optim import Adam
from torch import nn
from datetime import datetime
from utils.initialization import set_random_seed,args_print
from dataset.data import LabelConverter, MyDataLoader, MyDataset
from model.slu_bert import SimpleDecoder,get_output, TaggingFNNCRFDecoder
from utils.evaluator import Evaluator_bert as Evaluator
from utils.logger import Logger




install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)


from utils.args import init_args
from utils.initialization import set_random_seed, set_torch_device



# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")



if not args.dev_test:
    random_seeds = [99,999,9999,99999,114514]
    os.makedirs('trained-models', exist_ok=True)



    # prepare dataset & dataloader
    label_converter = LabelConverter('data/ontology.json')
    pretrained_model_name = 'bert-base-chinese'
    cache_dir = 'cache'

    print("begin initilization of bert")
    train_dataset = MyDataset(args,'data/train.json', label_converter, pretrained_model_name, cache_dir)
    dev_dataset = MyDataset(args,'data/development.json', label_converter, pretrained_model_name, cache_dir)
    print("initilization for bert")

    train_data_loader = MyDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_data_loader = MyDataLoader(dev_dataset)
    encoding_len = train_dataset[0][0][0].vector_with_noise.shape[1]
    print("initilizaion for dataset & dataloader")


    # logger information
    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f'bert.lr_{args.lr}.rnn_{args.rnn}.hidden_{args.hidden_size}.layer_{args.num_layer}.{datetime_now}'
    exp_dir = os.path.join('bert-result/', experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = Logger.init_logger(filename=exp_dir + '/Sche_LR--train.log')


    best_accuracies = []
    best_f1_scores = []
    best_precisions = []
    best_recalls = []

    print("finish initialization")

    for run, seed in enumerate(random_seeds):
        # model configuration
        set_random_seed(seed)

        # CRF
        if args.CRF:
            decoder = TaggingFNNCRFDecoder(args,encoding_len, label_converter.num_indexes).to(args.device)
        else:
            decoder = SimpleDecoder(args,encoding_len, label_converter.num_indexes).to(args.device)

        # check_point = torch.load(open('trained-models/slu-bert-LSTM-final.bin', 'rb'), map_location=args.device)
        # decoder.load_state_dict(check_point['model'])
        
        optimizer = Adam(decoder.parameters(), lr=0.01, weight_decay=1e-3)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

        loss_fn = nn.CrossEntropyLoss()

        best_acc = 0
        for epoch in range(400):
            #if epoch<100:
                #scheduler.step()
                #if epoch%20==0:
                #    print("epoch:",epoch)
                #    for param_group in optimizer.param_groups:
                #        print(f"Updated learning rate: {param_group['lr']}")
            
            if epoch==20:
                optimizer = Adam(decoder.parameters(), lr=0.001, weight_decay=0)

            # logger.info(f'Epoch: {epoch}')
            total_loss = 0
            # training
            decoder.train()
            for batch_x, batch_y in train_data_loader:
                batch_loss = 0
                for round_x, round_y in zip(batch_x, batch_y):
                    for x, y in zip(round_x, round_y):
                        output = decoder(x.vector_with_noise)
                        loss = loss_fn(output, y)
                        total_loss += loss
                        batch_loss += loss
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
            avg_loss = total_loss.item() / len(train_dataset)
            logger.info(f'Epoch: {epoch}  train. loss: {avg_loss}')

            # validation
            evaluator = Evaluator()
            decoder.eval()
            total_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in dev_data_loader:
                    for round_x, round_y in zip(batch_x, batch_y):
                        for x, y in zip(round_x, round_y):
                            input_vector = x.vector_with_noise if args.noise else x.vector_without_noise
                            output = decoder(input_vector)
                            loss = loss_fn(output, y)
                            total_loss += loss
                            input_tokens = x.tokens_with_noise if args.noise else x.tokens_without_noise
                            prediction = get_output(input_tokens, output, label_converter)
                            expected = get_output(x.tokens_with_noise, y, label_converter)
                            evaluator.add_result(prediction, expected)
            acc = evaluator.accuracy_rate
            pred,recall = evaluator.precision_rate, evaluator.recall_rate
            f1_score = evaluator.f1_score
            avg_loss = total_loss.item() / len(dev_dataset)
            logger.info(f'Acc: {acc:.5f}, F1 Score: {pred:.5f},{recall:.5f},{f1_score:.5f}, Avg. Loss: {avg_loss:.5f}')
            if acc > best_acc:
                # logger.info('New best!')
                best_acc = acc
                best_f1_score = f1_score
                best_precision = evaluator.precision_rate
                best_recall = evaluator.recall_rate
                torch.save({
                    'epoch': epoch,
                    'model': decoder.state_dict(),
                    'optim': optimizer.state_dict(),
                    'seed': seed,
                    'run': run,
                }, f'trained-models/slu-bert-{args.rnn}-{run}--Sche_LR.bin')
            
            if epoch%100==0:
                # Save the model checkpoint
                best_acc = acc
                best_f1_score = f1_score
                best_precision = evaluator.precision_rate
                best_recall = evaluator.recall_rate
                torch.save({
                    'epoch': epoch,
                    'model': decoder.state_dict(),
                    'optim': optimizer.state_dict(),
                    'seed': seed,
                    'run': run,
                }, f'trained-models/slu-bert-{args.rnn}-{run}--Sche_LR--{epoch}.bin')
        best_accuracies.append(best_acc)
        best_f1_scores.append(best_f1_score)
        best_precisions.append(best_precision)
        best_recalls.append(best_recall)

        
        


    print(best_accuracies)
    logger.info("Dev ACC:{:.4f}-+-{:.2f} Precision:{:.4f}-+-{:.2f} Recall:{:.4f}-+-{:.2f} F score:{:.4f}-+-{:.2f}".format(
        torch.tensor(best_accuracies).mean(), torch.tensor(best_accuracies).std(),
        torch.tensor(best_precisions).mean(), torch.tensor(best_precisions).std(),
        torch.tensor(best_recalls).mean(), torch.tensor(best_recalls).std(),
        torch.tensor(best_f1_scores).mean(), torch.tensor(best_f1_scores).std(),
    ))

# test on validation
elif args.dev_test:


    random_seeds = [99,999,9999,99999,114514]

    import sys, os, time, gc, json
    from utils.args import init_args
    from utils.initialization import *
    from utils.example import Example
    from utils.vocab import PAD
    from model.slu_baseline_tagging import SLUTagging
    args = init_args(sys.argv[1:])
    set_random_seed(args.seed)
    device = set_torch_device(args.device)

    start_time = time.time()
    train_path = "data/train.json"
    dev_path = "data/development.json"
    Example.configuration(args, train_path=train_path, word2vec_path=args.word2vec_path)
    train_dataset = Example.load_dataset(train_path)
    dev_dataset = Example.load_dataset(dev_path)
    print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
    print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

    args.vocab_size = Example.word_vocab.vocab_size
    args.pad_idx = Example.word_vocab[PAD]
    args.num_tags = Example.label_vocab.num_tags
    args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)


    model = SLUTagging(args).to(device)
    Example.word2vec.load_embeddings(model.word_embed, Example.word_vocab, device=device)


    # prepare dataset & dataloader
    label_converter = LabelConverter('data/ontology.json')
    pretrained_model_name = 'bert-base-chinese'
    cache_dir = 'cache'

    print("begin initilization of bert")
    train_dataset = MyDataset(args,'data/train.json', label_converter, pretrained_model_name, cache_dir)
    dev_dataset = MyDataset(args,'data/development.json', label_converter, pretrained_model_name, cache_dir)
    print("initilization for bert")
    
    train_data_loader = MyDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_data_loader = MyDataLoader(dev_dataset)
    encoding_len = train_dataset[0][0][0].vector_with_noise.shape[1]
    print("initilizaion for dataset & dataloader")


    best_accuracies = []
    best_f1_scores = []
    best_precisions = []
    best_recalls = []


    print("finish initialization")

    for run, seed in enumerate(random_seeds):
        # model configuration
        set_random_seed(seed)

        # crf
        if args.crf:
            decoder = TaggingFNNCRFDecoder(args,encoding_len, label_converter.num_indexes).to(args.device)
        else:
            decoder = SimpleDecoder(args,encoding_len, label_converter.num_indexes).to(args.device)


        check_point = torch.load(open('slu-bert-GRU-0--Sche_LR--300.bin', 'rb'), map_location='cuda')
        decoder.load_state_dict(check_point['model'])
        loss_fn = nn.CrossEntropyLoss()

        best_acc = 0
        for epoch in range(1):
            # validation
            evaluator = Evaluator()
            decoder.eval()
            total_loss = 0

            tmp_predictions = []
            tmp_expected = []

            with torch.no_grad():
                for batch_x, batch_y in dev_data_loader:
                    for round_x, round_y in zip(batch_x, batch_y):
                        for x, y in zip(round_x, round_y):
                            input_vector = x.vector_with_noise if args.noise else x.vector_without_noise
                            output = decoder(input_vector)
                            loss = loss_fn(output, y)
                            total_loss += loss
                            input_tokens = x.tokens_with_noise if args.noise else x.tokens_without_noise
                            prediction = get_output(input_tokens, output, label_converter)
                            expected = get_output(x.tokens_with_noise, y, label_converter)
                            tmp_predictions.append(prediction)
                            tmp_expected.append(expected)


            
            predictions = []
            expectations = []
            for i in range(len(tmp_predictions)):
                for j in range(len(tmp_predictions[i])):
                    predictions.append(tmp_predictions[i][j])

            if args.pinyin:
                pinyin_predictions = Example.pinyin_correction(predictions=predictions)
            else:
                pinyin_predictions = predictions

            for i in range(len(tmp_expected)):
                for j in range(len(tmp_expected[i])):
                    expectations.append(tmp_expected[i][j])

            pred_index = 0
            expectation_index = 0

            for i in range(len(tmp_expected)):
                evaluator.add_result(pinyin_predictions[pred_index:pred_index+len(tmp_predictions[i])], 
                expectations[expectation_index:expectation_index+len(tmp_expected[i])])
                pred_index+=len(tmp_predictions[i])
                expectation_index+=len(tmp_expected[i])
            
            
            acc = evaluator.accuracy_rate
            pred,recall = evaluator.precision_rate, evaluator.recall_rate
            f1_score = evaluator.f1_score
            avg_loss = total_loss.item() / len(dev_dataset)
            print(f'Acc: {acc:.5f}, F1 Score: {pred:.5f},{recall:.5f},{f1_score:.5f}, Avg. Loss: {avg_loss:.5f}')
            if acc > best_acc:
                best_acc = acc
                best_f1_score = f1_score
                best_precision = evaluator.precision_rate
                best_recall = evaluator.recall_rate

        best_accuracies.append(best_acc)
        best_f1_scores.append(best_f1_score)
        best_precisions.append(best_precision)
        best_recalls.append(best_recall)

    print(best_accuracies)
    print("Dev ACC:{:.4f}-+-{:.2f} Precision:{:.4f}-+-{:.2f} Recall:{:.4f}-+-{:.2f} F score:{:.4f}-+-{:.2f}".format(
        torch.tensor(best_accuracies).mean(), torch.tensor(best_accuracies).std(),
        torch.tensor(best_precisions).mean(), torch.tensor(best_precisions).std(),
        torch.tensor(best_recalls).mean(), torch.tensor(best_recalls).std(),
        torch.tensor(best_f1_scores).mean(), torch.tensor(best_f1_scores).std(),
    ))

